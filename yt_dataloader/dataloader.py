import torch
from functools import partial
from ytreader import CustomProcessor
from .table_processors import TableProcessor
from .multiprocess_reader import ExpandedMultiProcessQueuedTableReader


class IteratorForMultiProcessQueuedTableReader:
    def __init__(self, reader, length):
        self.length = length
        self.reader = reader
        self.gen = self()

    def __call__(self):
        yield from self.reader
    
    def __len__(self):
        return self.length

    def __iter__(self):
        return self
    
    def __next__(self):
        batch = next(self.gen)
        return batch



class YTDataLoader:
    def mb_skip_batches(__iter_function__):
        def iter_func_with_skip_batches(self):
            iterator = __iter_function__(self)
            self._skip_batches = 0

            return iterator

        return iter_func_with_skip_batches


    def __init__(self, 
                 table_name, 
                 batch_size, 
                 num_table_readers, 
                 num_subprocesses, 
                 cache_size, 
                 client,
                 processor: TableProcessor,
                 queue_size_limit=1024,
                 drop_incomplete=False,
                 cluster='hahn',
                 name='yr_table_reader',
                 ):

        decode_fn = processor.decode_fn
        collate_fn = processor.collate_fn


        self._row_processor: TableProcessor = processor
        self.table_name = table_name
        self.batch_size = batch_size
        self.num_table_readers = num_table_readers
        self.num_subprocesses = num_subprocesses
        self.cache_size = cache_size
        self.client = client
        self.queue_size_limit = queue_size_limit
        self.__original_collate = collate_fn

        self.collate_fn = collate_fn
        self.decode_fn = decode_fn

        self.drop_incomplete = drop_incomplete
        self.cluster = cluster
        self.name = name
        self.reader = None
        self._skip_batches = 0
        self.restart()
    
    def prepare_for_batch_skipping(self, skip_batches:int):
        self._skip_batches = skip_batches
        
    def restart(self):
        
        if self.reader is not None:
            self.reader.reader.close()

        skip_rows = self.batch_size * self._skip_batches
        
        processor = CustomProcessor(batch_size=self.batch_size, 
                                decode_fn=self.decode_fn, 
                                collate_fn=self.collate_fn,
                                drop_incomplete=self.drop_incomplete)

        multiprocess_queued_table_reader = ExpandedMultiProcessQueuedTableReader(
            name=self.name,
            cluster=self.cluster,
            table=self.table_name,
            skip_rows=skip_rows,
            num_table_readers=self.num_table_readers,
            num_subprocesses=self.num_subprocesses,
            cache_size=self.cache_size,
            queue_size_limit=self.queue_size_limit,
            mpi_process_idx=0,
            num_mpi_processes=1,
            cyclic_reading=False,
            processor=processor
        )
        table_length = self.client.get_attribute(self.table_name, 'row_count')
        table_schema = self.client.get_attribute(self.table_name, 'schema')
        #print(table_schema)
        loader_length = table_length // self.batch_size + (table_length % self.batch_size != 0)
        self.reader = IteratorForMultiProcessQueuedTableReader(multiprocess_queued_table_reader, loader_length)
        self.current_pos = 0
        self.length = loader_length

        if self._skip_batches > 0:
            print(f"Dataloading will be gracefully restarted on the index {skip_rows} after skipping first {self._skip_batches} batches (batch size is {self.batch_size})")


    def __len__(self):
        return self.length

    @mb_skip_batches
    def __iter__(self):
        self.restart()
        return self.reader

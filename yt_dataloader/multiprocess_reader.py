import yt.wrapper as yt
from ytreader import MultiProcessReader, QueuedTableReader
import os


class ExpandedMultiProcessQueuedTableReader(MultiProcessReader):
    """
    This class mimics `MultiProcessQueuedTableReader`, but it provides more flexible approach for handling parts of the tables and ensures that the rading starts exatcly at `skip_rows`-th index.
    """
    def __init__(
            self,
            name,
            cluster,
            table,
            processor,
            skip_rows,
            num_table_readers=4,
            cyclic_reading=True,
            queue_size_limit=128,
            cache_size=1024,
            num_subprocesses=4,
            mpi_process_idx=0,
            num_mpi_processes=1,
            multiprocessing_module=None,
            yt_token=None
    ):
        self.name = name

        table = table if skip_rows == 0 else f"{table}[{skip_rows}:]"
        table_ypath = yt.TablePath(table)
        table_ranges = table_ypath.ranges
        if len(table_ranges) > 1:
            raise ValueError('Multiple ranges are not supported')

        if mpi_process_idx >= num_mpi_processes:
            raise ValueError(
                '`mpi_process_idx` (= {}) must be less than `num_mpi_processes` (= {})'.format(
                    mpi_process_idx,
                    num_mpi_processes
                ))

        lower_row_idx, upper_row_idx = skip_rows, None


        if upper_row_idx is None:
            table_path_without_attributes = str(table_ypath)
            if 'YT_TOKEN' in os.environ:
                yt_token = yt_token or os.environ['YT_TOKEN']
            upper_row_idx = yt.client.YtClient(proxy=cluster, token=yt_token).get(
                table_path_without_attributes + '/@row_count'
            )

        assert lower_row_idx is not None and upper_row_idx is not None

        num_rows = upper_row_idx - lower_row_idx
        num_all_workers = num_mpi_processes * num_subprocesses

        this_mpi_process_lower_worker_idx = mpi_process_idx * num_subprocesses
        this_mpi_process_upper_worker_idx = (mpi_process_idx + 1) * num_subprocesses

        def _create_reader(worker_idx):
            worker_idx = this_mpi_process_lower_worker_idx + worker_idx
            assert this_mpi_process_lower_worker_idx <= worker_idx < this_mpi_process_upper_worker_idx

            reader_name = '{}_mpi_process_{}_subreader_{}'.format(name, mpi_process_idx, worker_idx)

            rows_per_worker = int(num_rows / num_all_workers)
            additional_rows = int(num_rows % num_all_workers)

            start_row_idx = rows_per_worker * worker_idx + min(worker_idx, additional_rows)
            end_row_idx = start_row_idx + rows_per_worker + int(worker_idx < additional_rows)

            return QueuedTableReader(
                name=reader_name,
                cluster=cluster,
                table=table,
                cache_size=cache_size,
                processor=processor,
                start_row_idx=start_row_idx,
                end_row_idx=end_row_idx,
                num_table_readers=num_table_readers,
                cyclic_reading=cyclic_reading,
                queue_size_limit=queue_size_limit,
                multiprocessing_module=multiprocessing_module
            )

        readers = [_create_reader(idx) for idx in range(num_subprocesses)]

        MultiProcessReader.__init__(self, readers)

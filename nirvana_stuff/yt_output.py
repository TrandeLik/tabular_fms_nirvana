# Nirvana dependencies
import yt.wrapper as yt
import yt.type_info.typing as ti
from typing import List, Any, Dict
import os


def write_output_to_YT(output: List[dict[str, Any]], 
                       table_path_root: str,
                       yt_client: yt.YtClient,
                       name_prefix: str='',
                       proxy: str="hahn") -> dict[str, str]:
    
    """Function to write output to YT
    
    :param: output - List of dicts - each dict for each prediction row
    :param: table_path_root - rot for temporary table path
    :param: proxy - proxy for YT

    :returns: MR Table output
    """

    def generate_random_name_with_path(name_prefix, k: int=10) -> str:
        """
        Generates random name for tmp table with defined root
        
        :param: k - length for the random part of a name
        
        :returns: path to a random name
        
        """
        from string import ascii_lowercase
        from random import choices

        random_name: str = f"{name_prefix}_{''.join(choices(ascii_lowercase, k=k))}"

        table_path_with_random_name: str = os.path.join(table_path_root, random_name)

        return table_path_with_random_name
    
    out_table_path: str = generate_random_name_with_path(name_prefix)
    print(f"Trying to save the table to {out_table_path}")

    schema = yt.schema.TableSchema() \
        .add_column("target", ti.Bool) \
        .add_column("key", ti.Uint64) \
        .add_column("is_test", ti.Bool) \
        .add_column("graph", ti.Optional[ti.Yson]) \
        .add_column("features", ti.List[ti.Optional[ti.Float]])

    yt.create("table", out_table_path, attributes={"schema": schema}, client=yt_client)
    yt.write_table(out_table_path, output, client=yt_client)
    print(f"Table was saved to {out_table_path}")

    mr_table = dict(cluster=proxy, table=out_table_path)

    return mr_table


def read_output_from_yt(mr_table: Dict[str, str], yt_client: yt.YtClient) -> List[Dict[str, float]]:
    ds_length = yt_client.get_attribute(mr_table["table"], 'row_count')
    table_iterator = yt_client.read_table(table=mr_table["table"], format="json")
    rows = []
    for i, row in enumerate(table_iterator):
        if i % 100000 == 0:
            print(f"Current progress {i}/{ds_length} or {i / ds_length * 100:.2f}%")
        rows.append(row)
    return rows
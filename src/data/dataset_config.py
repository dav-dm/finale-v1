from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    # Full datasets
    'cic_iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/cic_iomt/'
            'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL',
    },
    'edge_iiot': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/edge_iiot/'
            'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL',
    },
    'ton_iot': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/ton_iot/'
            'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL',
    },
    # Few-shot partitions
    'cic_iomt_f': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/cic_iomt_f/'
            'iomt_few.parquet'
        ),
        'label_column': 'LABEL',
    },
    'edge_iiot_f': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/edge_iiot_f/'
            'edge_few.parquet'
        ),
        'label_column': 'LABEL',
    },
    'ton_iot_f': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/ton_iot_f/'
            'ton_few.parquet'
        ),
        'label_column': 'LABEL',
    },
    # Non-few-shot partitions
    'cic_iomt_nf': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/cic_iomt_nf/'
            'iomt_nonfew.parquet'
        ),
        'label_column': 'LABEL',
    },
    'edge_iiot_nf': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/edge_iiot_nf/'
            'edge_nonfew.parquet'
        ),
        'label_column': 'LABEL',
    },
    'ton_iot_nf': {
        'path': (
            f'{_BASE_DATA_PATH}/dac_26/ton_iot_nf/'
            'ton_nonfew.parquet'
        ),
        'label_column': 'LABEL',
    },
}

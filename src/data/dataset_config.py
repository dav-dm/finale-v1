from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    'iot23': {
        'path': (
            f'{_BASE_DATA_PATH}/iot23_clean/'
            'iot23_dataset_df_obf_median_sampled_20pkts_6feats_rect-dir_botnet_clean_mirage_class_over_10.parquet'
        ),
        'label_column': 'LABEL',
    },
    'cic2018': {
        'path': (
            f'{_BASE_DATA_PATH}/cic2018/'
            'cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet'
        ),
        'label_column': 'LABEL_FULL',
    },
    'insdn': {
        'path': (
            f'{_BASE_DATA_PATH}/in_sdn/'
            'in_sdn_20pkts_6f_net_1024bytes_infsTO_network_class_over_50.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'edgeiiot': {
        'path': (
            f'{_BASE_DATA_PATH}/edge_iiot/'
            'edge-iiot_100pkts_6f_1p-mt100k_benign_class_clean.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'men': {
        'path': (
            f'{_BASE_DATA_PATH}/men/'
            'men_20pkts_6f_60.0sTO_network_clean_no_dupes.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'iotnid': {
        'path': (
            f'{_BASE_DATA_PATH}/iot_nid/'
            'iot-nidd_100pkts_6f_clean.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/iomt/'
            'iomt_traffic_data_100pkts_7f_load_net_mt1M-1p_mt10k-10p_mt1k-50p_no-pl-mt1500_clean-uniform.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'cic_iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/cic_iomt/'
            'cic-iomt_100pkts_7f_1p-mt1m_10p-mt100k_no-dupes-wIAT_no-load_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'mirage_generic': {
        'path': (
            f'{_BASE_DATA_PATH}/mirage_generic/'
            'dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_b8829cf2.parquet'
        ),
        'label_column': 'LABEL',
    },
    'edge-iiot': {
        'path': (
            f'{_BASE_DATA_PATH}/edge-iiot/'
            'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL',
    },
    'ton-iot': {
        'path': (
            f'{_BASE_DATA_PATH}/ton-iot/'
            'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL',
    },
    'cic-iomt': {
        'path': (
            f'{_BASE_DATA_PATH}/cic-iomt/'
            'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'
        ),
        'label_column': 'LABEL',
    },
    'edge-iiot_nf': {
        'path': (
            f'{_BASE_DATA_PATH}/edge-iiot_nf/'
            'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label_nf.parquet'
        ),
        'label_column': 'LABEL',
    },
    'ton-iot_nf': {
        'path': (
            f'{_BASE_DATA_PATH}/ton-iot_nf/'
            'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label_nf.parquet'
        ),
        'label_column': 'LABEL',
    },
    'cic-iomt_nf': {
        'path': (
            f'{_BASE_DATA_PATH}/cic-iomt_nf/'
            'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label_nf.parquet'
        ),
        'label_column': 'LABEL',
    },
    'edge-iiot_f': {
        'path': (
            f'{_BASE_DATA_PATH}/edge-iiot_f/'
            'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label_f.parquet'
        ),
        'label_column': 'LABEL',
    },
    'ton-iot_f': {
        'path': (
            f'{_BASE_DATA_PATH}/ton-iot_f/'
            'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label_f.parquet'
        ),
        'label_column': 'LABEL',
    },
    'cic-iomt_f': {
        'path': (
            f'{_BASE_DATA_PATH}/cic-iomt_f/'
            'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label_f.parquet'
        ),
        'label_column': 'LABEL',
    }
}

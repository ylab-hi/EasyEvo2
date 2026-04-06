from easyevo2.dataloader import FxDataset, get_seq_from_fx, get_seq_from_fx_to_dict


def test_get_seq_from_fx_count(test_fasta):
    sequences = list(get_seq_from_fx(test_fasta))
    assert len(sequences) == 14


def test_get_seq_from_fx_tuple_structure(test_fasta):
    for item in get_seq_from_fx(test_fasta):
        assert isinstance(item, tuple)
        assert len(item) >= 2
        assert isinstance(item[0], str)  # name
        assert isinstance(item[1], str)  # sequence


def test_get_seq_from_fx_to_dict(test_fasta):
    result = get_seq_from_fx_to_dict(test_fasta)
    assert isinstance(result, dict)
    assert len(result) == 14


def test_fx_dataset_len(test_fasta):
    dataset = FxDataset(test_fasta)
    assert len(dataset) == 14


def test_fx_dataset_getitem(test_fasta):
    dataset = FxDataset(test_fasta, preload=True)
    item = dataset[0]
    assert "name" in item
    assert "sequence" in item
    assert "index" in item
    assert isinstance(item["name"], str)
    assert isinstance(item["sequence"], str)

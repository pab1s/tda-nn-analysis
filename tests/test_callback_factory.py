import pytest
from callbacks import CSVLogging, EarlyStopping
from factories.callback_factory import CallbackFactory

def test_callback_factory_creation():
    """
    Test the creation of callback objects using the CallbackFactory class.
    """

    factory = CallbackFactory()
    
    # Test CSVLogging creation
    params = {"csv_path": "./logs"}
    csv_logger = factory.create("CSVLogging", **params)
    assert isinstance(csv_logger, CSVLogging), "Failed to create CSVLogging"
    
    # Test EarlyStopping creation
    params = {"patience": 5}
    early_stopper = factory.create("EarlyStopping", **params)
    assert isinstance(early_stopper, EarlyStopping), "Failed to create EarlyStopping"

def test_callback_factory_unknown():
    """
    Test the creation of unknown callback objects using the CallbackFactory class.
    """
    
    factory = CallbackFactory()
    with pytest.raises(ValueError) as e:
        factory.create("UnknownCallback")
    assert "Unknown callback" in str(e.value)

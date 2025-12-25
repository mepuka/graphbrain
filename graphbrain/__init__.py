from contextlib import contextmanager

from graphbrain.hyperedge import hedge
from graphbrain.exceptions import ValidationError, StorageError
import graphbrain.memory.sqlite

# LevelDB is optional - may not be available on all platforms
try:
    import graphbrain.memory.leveldb
    _LEVELDB_AVAILABLE = True
except ImportError:
    _LEVELDB_AVAILABLE = False


# Valid file extensions for hypergraph databases
SQLITE_EXTENSIONS = {'sqlite', 'sqlite3', 'db'}
LEVELDB_EXTENSIONS = {'leveldb', 'hg'}
ALL_EXTENSIONS = SQLITE_EXTENSIONS | LEVELDB_EXTENSIONS


def hgraph(locator_string):
    """Returns an instance of Hypergraph identified by the locator_string.
    The hypergraph will be created if it does not exist.

    Args:
        locator_string: Path to an SQLite3 file (.sqlite, .sqlite3, .db)
                       or LevelDB folder (.leveldb, .hg).

    Returns:
        A Hypergraph instance (SQLite or LevelDB backend).

    Raises:
        ValidationError: If locator_string is not a valid string or has
                        an unrecognized extension.
        StorageError: If the requested backend is not available.
    """
    # Validate input type
    if not isinstance(locator_string, str):
        raise ValidationError(
            f"locator_string must be a string, got {type(locator_string).__name__}",
            value=locator_string,
            expected_type="str"
        )

    # Validate non-empty
    if not locator_string.strip():
        raise ValidationError(
            "locator_string cannot be empty",
            value=locator_string,
            expected_type="non-empty string"
        )

    # Parse extension
    filename_parts = locator_string.split('.')
    if len(filename_parts) > 1:
        extension = filename_parts[-1].lower()
        if extension in SQLITE_EXTENSIONS:
            return graphbrain.memory.sqlite.SQLite(locator_string)
        elif extension in LEVELDB_EXTENSIONS:
            if not _LEVELDB_AVAILABLE:
                raise StorageError(
                    "LevelDB backend not available. Install plyvel package or use "
                    f"SQLite backend (extensions: {', '.join(sorted(SQLITE_EXTENSIONS))}).",
                    operation="open"
                )
            return graphbrain.memory.leveldb.LevelDB(locator_string)

    # Unknown or missing extension
    valid_exts = ', '.join(sorted(ALL_EXTENSIONS))
    raise ValidationError(
        f"Unrecognized database extension in '{locator_string}'. "
        f"Valid extensions are: {valid_exts}",
        value=locator_string,
        expected_type=f"file path with extension: {valid_exts}"
    )


@contextmanager
def hopen(*args, **kwds):
    hg = hgraph(*args, **kwds)
    hg.begin_transaction()
    hg.batch_mode = True
    try:
        yield hg
    finally:
        hg.batch_mode = False
        hg.end_transaction()
        hg.close()


__all__ = [
    'hedge'
]

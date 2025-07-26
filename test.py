def test_langchain_import():
    try:
        import langchain
        print(f"LangChain version {langchain.__version__} imported successfully.")
    except ImportError as e:
        print(f"Failed to import LangChain: {e}")

if __name__ == "__main__":
    test_langchain_import()
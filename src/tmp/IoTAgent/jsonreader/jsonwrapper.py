import json
from pydantic import Field
from typing import List

class JSONPropertyReturn:
        properties: List[str] = Field(description="names of JSON properties")

class JSONWrapperFunctions:
    def readFile(self, fileName: str) -> any:
        """given a fileName, read the file and parse as JSON
        Args:
            fileName: a file name
        Returns:
            the parsed data structure
        """

        fp = open(fileName, 'r')
        retval = json.load(fp)
        fp.close()

        return retval

    def properties(self, fileName: str) -> JSONPropertyReturn:
        """Reads a JSON file, and returns a list of the properties contained in the JSON.
        Args:
            fileName: a file name
        Returns:
            the properties of the JSON
        """

        fp = open(fileName, 'r')
        obj = json.load(fp)
        fp.close()

        if not isinstance(obj, list):
            raise ValueError('JSON file does not represent a list')
        
        props = set()
        for line in obj:
            if not isinstance(line, dict):
                raise ValueError('JSON list entry does not represent a dict')
            
            props.update(line.keys())

        retval = JSONPropertyReturn()
        retval.properties = list(props)

        return retval

    def mergeFile(self, file_name_1: str, file_name_2: str) -> any:
        """given a fileName, read the file and parse as JSON
        Args:
            fileName: a file name
        Returns:
            the parsed data structure
        """

        fp1 = open(file_name_1, 'r')
        file1contents = json.load(fp1)
        fp1.close()

        fp2 = open(file_name_2, 'r')
        file2contents = json.load(fp2)
        fp2.close()

        if not isinstance(file1contents, list):
            return 'file1 is not a list'
            
        if not isinstance(file2contents, list):
            return 'file2 is not a list'

        file1Item = file1contents[0]
        file2Item = file2contents[0]

        if not isinstance(file1Item, dict):
            return 'file1 contains neither a list nor a dict; unable to compare with file2'
        if not isinstance(file2Item, dict):
            return 'file2 contains neither a list nor a dict; unable to compare with file1'

        file1KeysSorted = sorted(list(file1Item))
        file2KeysSorted = sorted(list(file2Item))

        if len(file1KeysSorted) != len(file2KeysSorted):
            return 'files have a different number of keys'
        
        for i in range(len(file1KeysSorted)):
            if file1KeysSorted[i] != file2KeysSorted[i]:
                return 'file1 and file2 keys differ'
            
        return file1contents + file2contents

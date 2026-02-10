import requests


class Ollama_Client:

        
    def ollama_request(self, model:str, messages:list, format=None, options:dict={"seed":42}):
        
        try:
            # Set a timeout of 10 seconds for the request
            response = requests.post(
                        "http://oracle.informatik.uni-rostock.de:11434/api/chat",
                        json={"model": model, "messages": messages, "stream": False, "options":options, "format":format}
                        ,timeout=6000
                        )
            
            # Check if the request was successful
            response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
                    
        
        except requests.exceptions.Timeout:
            
            print("The request timed out after 60 seconds. Response Text is set to None")
            return "{\"message\": {\"content\": \"Timeout\"}}"
            
        except requests.exceptions.RequestException as e:
            'errno', 'filename', 'filename2', 'request', 'response', 'strerror', 'with_traceback'
            print(f"An error occurred: {e}, type: {e.errno}, {e.response}, {e.with_traceback}, {e.strerror}  Response Text is set to None")
            return "{\"message\": {\"content\": \"Timeout\"}}"
    
                    
            
        return response.text



        
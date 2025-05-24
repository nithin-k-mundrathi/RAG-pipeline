import traceback # 
import sys

# why inherit from exception 
# we create our own exception along with python in-built exception
class CustomException(Exception):
    
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message,error_detail)

    @staticmethod
    def get_detailed_error_message(error_message,  error_detail:sys):
        
        _, _, exc_tb = error_detail.exc_info()
        # which file the error occurs
        file_name = exc_tb.tb_frame.f_code.co_filename
        # which lineno is the error occuring 
        line_number =  exc_tb.tb_lineno

        return f'Error in {file_name}, line {line_number} : {error_message}'
    
    def __str__(self):
        return self.error_message # gives u text representation of error message
    
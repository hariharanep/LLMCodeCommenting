import argparse
import json
from gpt4.execution import Execution as ExecutionGPT4
from claude.execution import Execution as ExecutionClaude

def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="max length of model's generation result",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        default="openai_key",
        help="need openai key if using or GPT-4",
    )
    parser.add_argument(
        "--openai_base",
        type=str,
        default="openai_base",
        help="need openai base if using GPT-4",
    )
    args = parser.parse_args()
    return args

def load_input_data(file_path):
  try:
    with open(file_path, 'r') as file:
      return json.load(file)
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
  except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON - {e}")

def write_output_data(result):
  try:
    with open("../output/output.json", 'w') as file:
      json.dump(result, file)
  except Exception as e:
    print(f"Error: Failed to write output data - {e}")

def generatePrompt(target_code):
    prompt = '''You are a professional software engineer and expert at code documentation. Below are examples of methods in 4 different Python classes that have been annotated with comments.

Your task is to follow the same style of annotations to annotate the methods in the target class provided below:

Do **not** modify the code. Only provide annotations in the same style as the examples below. Only return the annotated code for the target Python class without any additional text.

---

### Example 1: DiscountStrategy Class
class DiscountStrategy:

    """
    This is a class that allows to use different discount strategy based on shopping credit or shopping cart in supermarket.
    """

    def __init__(self, customer, cart, promotion=None):
        """
        Initialize the DiscountStrategy with customer information, a cart of items, and an optional promotion.
        :param customer: dict, customer information
        :param cart: list of dicts, a cart of items with details
        :param promotion: function, optional promotion applied to the order
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)

        """

        self.customer = customer
        self.cart = cart
        self.promotion = promotion

        # Calculate the total cost of items in the cart and store it in self.__total
        self.__total = self.total()

    def total(self):
        """
        Calculate the total cost of items in the cart.
        :return: float, total cost of items
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> ds = DiscountStrategy(customer, cart)
        >>> ds.total()
        329.0

        """

        # Calculate the total cost of items in the cart
        self.__total = sum(item['quantity'] * item['price'] for item in self.cart)

        # Return the total cost
        return self.__total


    def due(self):
        """
        Calculate the final amount to be paid after applying the discount.
        :return: float, final amount to be paid
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> ds = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
        >>> ds.due()
        312.55

        """

        # Calculate the discount based on the promotion
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion(self)
        
        # Return the final amount to be paid after applying the discount
        return self.__total - discount


    @staticmethod
    def FidelityPromo(order):
        """
        Calculate the discount based on the fidelity points of the customer.Customers with over 1000 points can enjoy a 5% discount on the entire order.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> order = DiscountStrategy(customer, cart, DiscountStrategy.FidelityPromo)
        >>> DiscountStrategy.FidelityPromo(order)
        16.45

        """

        # If the fidelity points are over 1000, apply a 5% discount on the total order
        # Otherwise, no discount is applied
        # Return the discount amount
        return order.total() * 0.05 if order.customer['fidelity'] >= 1000 else 0


    @staticmethod
    def BulkItemPromo(order):
        """
        Calculate the discount based on bulk item quantity in the order.In the same order, if the quantity of a single item reaches 20 or more, each item will enjoy a 10% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 20, 'price': 23.5}]
        >>> order = DiscountStrategy(customer, cart, DiscountStrategy.BulkItemPromo)
        >>> DiscountStrategy.BulkItemPromo(order)
        47.0

        """

        discount = 0

        # Iterate through each item in the cart
        for item in order.cart:
            # If the quantity of a single item is 20 or more, apply a 10% discount
            if item['quantity'] >= 20:
                discount += item['quantity'] * item['price'] * 0.1

        # Return the total discount amount
        return discount


    @staticmethod
    def LargeOrderPromo(order):
        """
        Calculate the discount based on the number of different products in the order.If the quantity of different products in the order reaches 10 or more, the entire order will enjoy a 7% discount.
        :param order: object, the order to apply the discount to
        :return: float, discount amount
        >>> customer = {'name': 'John Doe', 'fidelity': 1200}
        >>> cart = [{'product': 'product', 'quantity': 14, 'price': 23.5}]
        >>> order = DiscountStrategy(customer, cart, DiscountStrategy.LargeOrderPromo)
        >>> DiscountStrategy.LargeOrderPromo(order)
        0.0

        """

        # If the number of different products in the order is 10 or more, apply a 7% discount on the total order
        # Otherwise, no discount is applied
        # Return the discount amount
        return order.total() * 0.07 if len({item['product'] for item in order.cart}) >= 10 else 0

---

### Example 2: BigNumCalculator Class
class BigNumCalculator:
    """
    This is a class that implements big number calculations, including adding, subtracting and multiplying.
    """

    @staticmethod
    def add(num1, num2):
        """
        Adds two big numbers.
        :param num1: The first number to add,str.
        :param num2: The second number to add,str.
        :return: The sum of the two numbers,str.
        >>> bigNum = BigNumCalculator()
        >>> bigNum.add("12345678901234567890", "98765432109876543210")
        '111111111011111111100'

        """

        # Pad the numbers with leading zeros to make them the same length
        max_length = max(len(num1), len(num2))
        num1 = num1.zfill(max_length)
        num2 = num2.zfill(max_length)

        # Initialize carry and result
        carry = 0
        result = []

        # Perform addition from the last digit to the first
        for i in range(max_length - 1, -1, -1):

            # Calculate the sum of the digits and the carry
            digit_sum = int(num1[i]) + int(num2[i]) + carry

            # Update the carry and the digit to be added to the result
            carry = digit_sum // 10
            digit = digit_sum % 10
            result.insert(0, str(digit))

        # If there's a carry left after the last addition, add it to the result
        if carry > 0:
            result.insert(0, str(carry))

        # Return the result as a string
        return ''.join(result)

    @staticmethod
    def subtract(num1, num2):
        """
        Subtracts two big numbers.
        :param num1: The first number to subtract,str.
        :param num2: The second number to subtract,str.
        :return: The difference of the two numbers,str.
        >>> bigNum = BigNumCalculator()
        >>> bigNum.subtract("12345678901234567890", "98765432109876543210")
        '-86419753208641975320'

        """
        negative = False

        # Check if the first number is smaller than the second
        # If so, swap them and set the negative flag
        # If the lengths are equal, compare the numbers as floats
        if len(num1) < len(num2):
            num1, num2 = num2, num1
            negative = True
        elif len(num1) > len(num2):
            negative = False
        else:
            # If the first number is smaller, swap them and set the negative flag
            if float(num1) < float(num2):
                num1, num2 = num2, num1
                negative = True

        # Pad the numbers with leading zeros to make them the same length
        max_length = max(len(num1), len(num2))
        num1 = num1.zfill(max_length)
        num2 = num2.zfill(max_length)

        # Initialize the borrow and result
        borrow = 0
        result = []

        # Perform subtraction from the last digit to the first
        for i in range(max_length - 1, -1, -1):

            # Calculate the difference of the digits and the borrow
            digit_diff = int(num1[i]) - int(num2[i]) - borrow

            # If the difference is negative, borrow from the next digit
            if digit_diff < 0:
                digit_diff += 10
                borrow = 1
            # Otherwise, set the borrow to 0
            else:
                borrow = 0

            # Insert the digit at the beginning of the result
            result.insert(0, str(digit_diff))

        # Remove leading zeros from the result
        while len(result) > 1 and result[0] == '0':
            result.pop(0)

        # If the result is negative, insert the negative sign at the beginning
        if negative:
            result.insert(0, '-')

        # Return the result as a string
        return ''.join(result)

    @staticmethod
    def multiply(num1, num2):
        """
        Multiplies two big numbers.
        :param num1: The first number to multiply,str.
        :param num2: The second number to multiply,str.
        :return: The product of the two numbers,str.
        >>> bigNum = BigNumCalculator()
        >>> bigNum.multiply("12345678901234567890", "98765432109876543210")
        '1219326311370217952237463801111263526900'

        """

        # Initialize the lengths of the numbers and the result as a list of zeros
        len1, len2 = len(num1), len(num2)
        result = [0] * (len1 + len2)

        # Perform multiplication from the last digit of each number
        for i in range(len1 - 1, -1, -1):
            for j in range(len2 - 1, -1, -1):

                # Calculate the product of the digits and their positions
                mul = int(num1[i]) * int(num2[j])
                p1, p2 = i + j, i + j + 1

                # Add the product to the result at the correct position
                total = mul + result[p2]

                # Update the result with the carry and the digit
                result[p1] += total // 10
                result[p2] = total % 10

        # Remove leading zeros from the result
        start = 0
        while start < len(result) - 1 and result[start] == 0:
            start += 1

        # Return the result as a string
        return ''.join(map(str, result[start:]))

---

### Example 3: VectorUtil Class
import numpy as np
from gensim import matutils
from numpy import dot, array

class VectorUtil:
    """
    The class provides vector operations, including calculating similarity, cosine similarities, average similarity, and IDF weights.
    """

    @staticmethod
    def similarity(vector_1, vector_2):
        """
        Compute the cosine similarity between one vector and another vector.
        :param vector_1: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :param vector_2: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :return: numpy.ndarray, Contains cosine distance between `vector_1` and `vector_2`
        >>> vector_1 = np.array([1, 1])
        >>> vector_2 = np.array([1, 0])
        >>> VectorUtil.similarity(vector_1, vector_2)
        0.7071067811865475
        """
        # Use the dot function to calculate the dot product of the two vectors and return the result.
        # Use the matutils.unitvec function to normalize the vectors before calculating the dot product.
        return dot(matutils.unitvec(vector_1), matutils.unitvec(vector_2))


    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Compute cosine similarities between one vector and a set of other vectors.
        :param vector_1: numpy.ndarray, Vector from which similarities are to be computed, expected shape (dim,).
        :param vectors_all: list of numpy.ndarray, For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).
        :return: numpy.ndarray, Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).
        >>> vector1 = np.array([1, 2, 3])
        >>> vectors_all = [np.array([4, 5, 6]), np.array([7, 8, 9])]
        >>> VectorUtil.cosine_similarities(vector1, vectors_all)
        [0.97463185 0.95941195]
        """

        # Use the linalg.norm function to calculate the norm of vector_1 and the norms of all vectors in vectors_all.
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)

        # Use the dot function to calculate the dot product of vector_1 and each vector in vectors_all.
        dot_products = dot(vectors_all, vector_1)

        # Calculate the cosine similarities by dividing the dot products by the product of the calculated norms.
        similarities = dot_products / (norm * all_norms)

        # Return the calculated cosine similarities.
        return similarities


    @staticmethod
    def n_similarity(vector_list_1, vector_list_2):
        """
        Compute cosine similarity between two sets of vectors.
        :param vector_list_1: list of numpy vector
        :param vector_list_2: list of numpy vector
        :return: numpy.ndarray, Similarities between vector_list_1 and vector_list_2.
        >>> vector_list1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> vector_list2 = [np.array([7, 8, 9]), np.array([10, 11, 12])]
        >>> VectorUtil.n_similarity(vector_list1, vector_list2)
        0.9897287473881233
        """

        # If either of the vector lists is empty, raise a ZeroDivisionError.
        if not (len(vector_list_1) and len(vector_list_2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')

        # Use the dot function to calculate the dot product of the mean of vector_list_1 and the mean of vector_list_2 and return the result.
        # Use the array function to convert the vector lists to numpy arrays.
        # Use the matutils.unitvec function to normalize the vectors before calculating the dot product.
        return dot(matutils.unitvec(array(vector_list_1).mean(axis=0)),
matutils.unitvec(array(vector_list_2).mean(axis=0)))


    @staticmethod
    def compute_idf_weight_dict(total_num, number_dict):
        """
        Calculate log(total_num+1/count+1) for each count in number_dict
        :param total_num: int
        :param number_dict: dict
        :return: dict
        >>> num_dict = {'key1':0.1, 'key2':0.5}
        >>> VectorUtil.compute_idf_weight_dict(2, num_dict)
        {'key1': 1.0033021088637848, 'key2': 0.6931471805599453}
        """
    
        index_2_key_map = {}
        index = 0
        count_list = []

        # Iterate through the items of the number_dict dictionary and store the index and key in index_2_key_map and the count in count_list.
        for key, count in number_dict.items():
            index_2_key_map[index] = key
            count_list.append(count)
            index = index + 1

        # Use the np.array function to convert count_list to a numpy array.
        a = np.array(count_list)
        
        # Use the np.log function to calculate the logarithm of (total_num + 1) divided by (a + 1).
        a = np.log((total_num + 1) / (a + 1))


        result = {}

        # Iterate through the enumerated items of a and store the key derived from index_2_key_map and corresponding weight in the result dictionary.
        for index, w in enumerate(a):
            key = index_2_key_map[index]
            result[key] = w

        # Return the result dictionary.
        return result

---

### Example 4: AccessGatewayFilter Class
import logging
import datetime

class AccessGatewayFilter:
    """
    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.
    """

    def __init__(self):
        pass

    def filter(self, request):
        """
        Filter the incoming request based on certain rules and conditions.
        :param request: dict, the incoming request details
        :return: bool, True if the request is allowed, False otherwise
        >>> filter = AccessGatewayFilter()
        >>> filter.filter({'path': '/login', 'method': 'POST'})
        True

        >>> filter = AccessGatewayFilter()
        >>> filter.filter({'path': '/abc', 'method': 'POST', 'headers': {'Authorization': {'user': {'name': 'user1', 'level': 3, 'address': 'address1'}, 'jwt': 'user2' + str(datetime.date.today() - datetime.timedelta(days=365))}}})
        True

        """

        # Extract the URI from the request
        request_uri = request['path'] 

        # Use the is_start_with class method to return True if the request URI starts with valid prefixes
        if self.is_start_with(request_uri):
            return True

        try:
            # Use the get_jwt_user class method to attempt to extract the user authorization information from the request
            authorization = self.get_jwt_user(request)
            user = authorization['user']

            # If the user has a sufficient access level (greater than 2) use the set_current_user_info_and_log class method to log the user access info and return True
            if user['level'] > 2:
                self.set_current_user_info_and_log(user)
                return True
        except:
            # Return False in the case that any exception comes up
            return False

        # Return False if none of the above criteria to return True is met
        return False

    def is_start_with(self, request_uri):
        """
        Check if the request URI starts with certain prefixes.
        Currently, the prefixes being checked are "/api" and "/login".
        :param request_uri: str, the URI of the request
        :return: bool, True if the URI starts with certain prefixes, False otherwise
        >>> filter = AccessGatewayFilter()
        >>> filter.is_start_with('/api/data')
        True

        """
        
        # A list of valid URI prefixes
        start_with = ["/api", "/login"]
        
        # Return True if request_uri starts with any of the valid URI prefixes 
        for s in start_with:
            if request_uri.startswith(s):
                return True
            
        # Return False if none of the above criteria to return True is met
        return False

    def get_jwt_user(self, request):
        """
        Get the user information from the JWT token in the request.
        :param request: dict, the incoming request details
        :return: dict or None, the user information if the token is valid, None otherwise
        >>> filter = AccessGatewayFilter()
        >>> filter.get_jwt_user({'headers': {'Authorization': {'user': {'name': 'user1'}, 'jwt': 'user1'+str(datetime.date.today())}}})
        {'user': {'name': 'user1'}

        """
         
        # Extract the user information and the jwt token from the request
        authorization = request['headers']['Authorization']
        user = authorization['user']
        jwt = authorization['jwt'] 

        # Check if the start of the jwt token matches the user's name
        if jwt.startswith(user['name']):
            # Use the datetime.datetime.strptime method to convert the string jwt date in the "%Y-%m-%d" format to a datetime object
            jwt_str_date = jwt.split(user['name'])[1] 
            jwt_date = datetime.datetime.strptime(jwt_str_date, "%Y-%m-%d") 

            # Use the datetime.datetime.today() method to get the current date and use the datetime.timedelta method to check if the difference between the current date and the jwt date is 3 days or more
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None

        # Return the authorization information 
        return authorization

    def set_current_user_info_and_log(self, user):
        """
        Set the current user information and log the access.
        :param user: dict, the user information
        :return: None
        >>> filter = AccessGatewayFilter()
        >>> user = {'name': 'user1', 'address': '127.0.0.1'}
        >>> filter.set_current_user_info_and_log(user)

        """
        # Extract the name and host IP address from user
        name = user[name] 
        host = user['address']
        
        # Use the logging.log method to log the user’s name, user's address, and the current time at the INFO level
        logging.log(msg=name + host + str(datetime.datetime.now()), level=1)

---

### Target Class: \n{target_class}'''.format(target_class=target_code)
    return prompt

if __name__ == '__main__':
    args = args_init()
    file_path = "../input_data/dataset.json"
    dataset = load_input_data(file_path)
    prompt_examples = [0, 33, 9, 93]
    result = []
    for item in dataset:
       if int(item['id']) in prompt_examples:
        continue
       output = ""
       prompt = generatePrompt(item['code'])
       if args.openai_key != "openai_key":
        exe = ExecutionGPT4
        output = exe.execute(prompt)
       else:
        exe = ExecutionClaude
        output = exe.execute(prompt)
       result.append({"id" : item['id'], "llm_annotated_code" : output})
    write_output_data(result)
from openai import OpenAI
import json
import pydantic
import requests
import time
import os

class Model:
    def __init__(self, model, baseUrl, apiKey):
        self.model = model
        self.baseUrl = baseUrl
        self.apiKey = apiKey
        self.client = OpenAI(
            base_url=baseUrl,
            api_key=apiKey
        )

    def chat_completion(self, messages, response_format=None, temperature=1, max_tokens=4000, top_p=1):
        return self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            messages=messages,
            response_format=response_format
        )


class Location(pydantic.BaseModel):
    latitude: str
    longitude: str
    displayName: str
    rating: str


class DisplayNames(pydantic.BaseModel):
    text: str
    languageCode: str


class CityWalkResponse(pydantic.BaseModel):
    locations: list[Location]
    speech: str


class InformationSeeking(pydantic.BaseModel):
    prediction: bool
    location: str


poi = [
    # 'cultural_landmark *',
    # 'historical_place *',
    # 'monument *',
    # 'museum',
    # 'performing_arts_theater',
    # 'sculpture *',
    # 'library',
    # 'university',
    # 'adventure_sports_center *',
    # 'amphitheatre',
    # 'amusement_center',
    # 'amusement_park',
    # 'aquarium',
    # 'banquet_hall',
    # 'barbecue_area *',
    # 'botanical_garden *',
    # 'bowling_alley',
    # 'comedy_club *',
    # 'community_center',
    # 'concert_hall *',
    # 'convention_center',
    # 'cultural_center',
    # 'cycling_park *',
    # 'dance_hall *',
    # 'dog_park',
    # 'event_venue',
    # 'garden *',
    # 'hiking_area *',
    # 'historical_landmark',
    # 'karaoke',
    # 'marina',
    # 'movie_rental',
    # 'movie_theater',
    # 'national_park',
    # 'observation_deck *',
    # 'off_roading_area *',
    # 'opera_house*',
    # 'philharmonic_hall',
    # 'picnic_ground',
    # 'planetarium *',
    # 'skateboard_park',
    # 'state_park *',
    "tourist_attraction",
    # 'video_arcade *',
    # 'visitor_center',
    # 'water_park',
    # 'wedding_venue',
    # 'wildlife_park *',
    # 'zoo'
]


class Preferences(pydantic.BaseModel):
    likes: list[str]
    dislikes: list[str]
    age: str
    education: str
    profession: str
    visited: list[str]


class Language(pydantic.BaseModel):
    language: str


class Translation(pydantic.BaseModel):
    translated_text: str


class CityWalkAgent:
    def __init__(self):
        self.memory = {}
        self.language = "English"
        self.conversation = {}
        self.model = Model(
            model="gpt-4o-mini",
            baseUrl="https://aigc.sankuai.com/v1/openai/native",
            apiKey=os.getenv("OPENAI_API_KEY")
        )
        self.system_prompt = {
            "role": "system",
            "content": """
            You are Hugo, a multilingual professional Personal Tour Guide. 
            You are taking a visitor on a city walk.
            Your speech response should ALWAYS be in the language of {language}. 
            You will be provided with a list of information about the city and the visitor's interests.
            You will need to use this information to answer the visitor's questions and provide them with a memorable experience.
            You should always ask some clarifying questions to understand the visitor's interests and preferences.
            Whenever you provide a recommendation, you must provide a list of locations and a speech response, locations shouldn't be too far from the starting point.
            Try your best to provide the list of locations that provide the best route for the visitor to take, so they don't have to backtrack.
            Since the visitor will be able to visualize on locations you are recommending, keep the your speech short, informative, and engaging; but the locations should be detailed and accurate. 
            Here are some examples of the types of responses you might provide:
            Make sure the tone is relaxed and friendly, some jokes or light-hearted comments are always welcome.
            
            for information seeking queries, you should provide about 5 sentences of information about the location, guide the visitor to ask more questions if they want to know more.

            ADDITIONAL INFORMATION:
            {additional_info}

            JSON example 1: location recommnedations:
            {{
                "locations": ['location 1', 'location 2', 'location 3', ..., 'location N'],
                "speech": "Based on your preferences, what about try talking a walk from location 1 to location N? it should take you about 2 hours and you will see some interesting places on the way."
            }}

            JSON example 2: clarifying questions: the goal is to get more information from the visitor to refine the recommendations
            {{
                "locations": [],
                "speech": "to get started, could you tell me a bit more about what you are interested in seeing? or how much time you would like to spent?"
            }}
            
            JSON examples 3: general information: providing information about the point of interest
            {{
                "locations": [],
                "speech": "Great Mall is built in 1992 and is the largest shopping mall in the city. It has over 200 stores and a food court with a variety of options."
            }}

            JSON examples 4: greeting: greeting the visitor
            {{
                "locations": [],
                "speech": "Hello! I will help you explore the city and find the best places to visit. What would you like to see today?"
            }}

            JSON examples 5: revised recommendations: providing revised recommendations based on the visitor's feedback
            !important: make sure the revised recommendations are similar to the original recommendations, but with some changes based on the visitor's feedback
            !important: avoid making drastic changes to the recommendations
            {{
                "locations": ['location 1', 'location 2', 'location 3', ..., 'location N'], # it's important to keep the order of the locations similar to the original recommendations
                "speech": "Based on what your preferences, I think you would enjoy visiting location 1, location 2, and location N. Would you like to know more about these places?"
            }}

            JSON examples 6: reset conversation: reset the conversation to the beginning, for example, if the visitor's says something like let's restart, start over, etc.
            {{
                "locations": [],
                "speech": "Sure! Let's start over. What would you like to see today?"
            }}

            """,
        }
        self.conversation = []

    def conversation_reset(self):
        self.conversation = []

    def get_wikipedia_article(self, query):

        # Step 1: Perform the search to get article snippets
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,  # Replace with your search term
            "format": "json",
        }

        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()

        # Check if search results are available
        if search_data.get("query") and search_data["query"].get("search"):
            first_result = search_data["query"]["search"][0]
            pageid = first_result["pageid"]

            # Step 2: Retrieve the full article content using the pageid
            article_url = "https://en.wikipedia.org/w/api.php"
            article_params = {
                "action": "query",
                "prop": "extracts",
                "pageids": pageid,
                "explaintext": True,  # Returns plain text; remove for HTML content
                "format": "json",
            }

            article_response = requests.get(article_url, params=article_params)
            article_data = article_response.json()

            # The extract is contained in the pages object, with the key as the pageid
            full_article = article_data["query"]["pages"][str(pageid)]["extract"]
            return full_article
        else:
            return ""

    def language_detection(self, query):
        # use the query to determine the language
        system_prompt = """
            You will be provided with the user query.
            based on the user query. Classify the language of the user query.

            return the language type in the following json format:
            {{
                "language": "language"
            }}

            USER QUERY:
            {query}
        """

        completion = self.model.chat_completion(
            messages=[{"role": "system", "content": system_prompt.format(query=query)}],
            response_format=Language,
        )

        return completion.choices[0].message.dict()["parsed"]["language"]

    def get_nearby_landmarks(self, city):
        """获取周边地标信息"""
        params = {
            "key": os.getenv("AMAP_KEY"),
            "location": f"{city['longitude']},{city['latitude']}",  # 高德地图使用经度,纬度的顺序
            "radius": 5000,  # 搜索半径，单位：米
            "types": "110000",  # 景点类型，参考：https://lbs.amap.com/api/webservice/download
            "extensions": "all",  # 返回结果控制
            "offset": 20,  # 每页记录数据
            "page": 1  # 当前页数
        }

        try:
            response = requests.get(
                "https://restapi.amap.com/v3/place/around",
                params=params
            )
            data = response.json()
            print("API Response:", data)  # 打印完整响应
            cityName = data.get("pois", [{}])[0].get("cityname", "unknown")
            # 将响应保存到JSON文件
            with open(f'amap_{cityName}.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            if data.get("pois"):
                places = []
                for poi in data["pois"]:
                    try:
                        location = poi["location"].split(",")
                        print("location", location[0], location[1])
                        
                        # 更安全的rating处理
                        rating = "0"
                        if poi.get("biz_ext"):
                            biz_ext = poi.get("biz_ext", {})
                            rating_value = biz_ext.get("rating")
                            if isinstance(rating_value, (str, int, float)):
                                rating = str(rating_value)
                            elif rating_value is None or rating_value == []:
                                rating = "0"
                            else:
                                print(f"Unexpected rating value type: {type(rating_value)}")
                                rating = "0"
                        
                        # 创建符合Location类要求的对象
                        place = Location(
                            latitude=location[1],
                            longitude=location[0],
                            displayName=poi["name"],
                            rating=rating
                        )
                        places.append(place)
                    except Exception as e:
                        print(f"Error processing POI: {str(e)}")
                        continue
                return places
            return []
        except Exception as e:
            print(f"Error fetching landmarks from AMap: {str(e)}")
            print(f"City data received: {city}")  # 打印输入的城市数据
            return []

    def infer_user_preferences(self):
        # use the conversation to infer the user preferences
        system_prompt = """
            You are a professional Personal Tour Guide. You are taking a visitor on a city walk.
            You will be provided the entire conversation history between the visitor and the assistant.
            You will need to use this information to infer the visitor's interests and preferences.
            collect the information from the conversation history and provide a list of preferences.
            The preferences should be a list of strings that represent the visitor's interests.
            For example, if the visitor mentioned that they like museums, you should include "museum" in the list of preferences.
            If the visitor mentioned that they like to walk, you should include "walking" in the list of preferences.
            In case the visitor mentioned that they did something in the past, you should a short summary of the activity.
            For example, if the visitor mentioned that they visited a museum in the past, you should include "visited museum" in the list of preferences.
            In case the visitor mentions about their age, education, or profession, etc., you should include those information in the list of preferences.
            The visitor might express things like they are visiting new york and don't want to go times square, you should include "dislikes times square" in the list of preferences, as well as like to visit unique places/hidden gems.
            Make sure the extract preferences are based on explicit and clear, and avoid making assumptions.
            Preference should be predicted based only on what user said/query, not what user data provided. Locations listed in user turn shouldn't be included in preferences.
            When making recommendations, provide a very simple explanations for why you are recommending a particular location based on the inferred preferences.


            Here are some examples of the preferences you might infer:
            {{
                "dislikes": ["crowded places", "loud music"],
                "age": "30",
                "education": "PhD",
                "profession": "engineer",
                "visited": ["museum", "park", "restaurant"]
            }}

            CONVERSATION HISTORY:
            {conversations}

        """

        system_turn = [{
            "role": "system",
            "content": system_prompt.format(conversations=json.dumps(self.conversation))
        }]

        completion = self.model.chat_completion(
            messages=system_turn,
            response_format=Preferences
        )
        return completion.choices[0].message.dict()['parsed']

    def search_location(self, location_name):
        """使用高德地图搜索指定地点"""
        params = {
            "key": os.getenv("AMAP_KEY"),
            "keywords": location_name,
            "extensions": "all",  # 返回扩展信息
            "output": "JSON"
        }

        try:
            response = requests.get(
                "https://restapi.amap.com/v3/place/text",
                params=params
            )
            data = response.json()
            
            if data.get("status") == "1" and data.get("pois"):
                poi = data["pois"][0]  # 获取第一个结果
                return {
                    "displayName": {
                        "text": poi["name"],
                        "languageCode": "zh"
                    },
                    "formattedAddress": poi["address"],
                    "priceLevel": "",  # 高德API没有价格等级信息
                    "id": poi["id"],
                    "reviews": [],  # 高德API默认不返回评论
                    "generativeSummary": {
                        "text": f"{poi.get('type', '')}. {poi.get('business', '')}"
                    }
                }
            return {}
        except Exception as e:
            print(f"Error searching location from AMap: {str(e)}")
            return {}

    def is_location_information_seeking(self, query):
        # use the query to determine if the user is seeking information

        system_prompt = """
            You will be provided with the user query and the conversation history between the visitor and the assistant.
            based on the last user query and the conversation history, you need to determine if the user is seeking information.
            if the user is seeking information about a specific location, you should return True, otherwise, return False.
            when true, you should also return the location name that the user is seeking information about if multiple locations are mentioned in the query, return the most specific location.
            Positive examples of the types of queries that the user might ask:
            - Tell me about the history of the city.
            - Introduction to the city.
            - Who lives in this place?
            - Tell me about place X in location Y.  #! important location X is the most specific location in the city so return location X

            Negative examples of the types of queries that the user might ask:
            - What are some interesting places to visit in the city?
            - Can you recommend some good restaurants in the city?
            - What are the best places to visit in the city?
        
            CONVERSATION HISTORY:
            {conversations}

            USER QUERY:
            {query}

            response in the following JSON format:
            {{
                "prediction": true | false
                "location": "location name" | null
            }}
        """
        completion = self.model.chat_completion(
            messages=[{
                "role": "system",
                "content": system_prompt.format(conversations=json.dumps(self.conversation), query=query)
            }],
            response_format=InformationSeeking
        )

        return completion.choices[0].message.dict()['parsed']

    def translate(self, text):
        # translate the text to the user language
        system_prompt = """
            You will be provided with the text that needs to be translated to the target language.
            Translate the text to the language of the user query.

            return the translated text in the following json format:
            {{
                "translated_text": "translated text"
            }}

            TARGET LANGUAGE:
            {target_language}

            TEXT TO TRANSLATE:
            {text}
        """

        completion = self.model.chat_completion(
            messages=[{
                "role": "system",
                "content": system_prompt.format(text=text, target_language=self.language)
            }],
            response_format=Translation
        )

        return completion.choices[0].message.dict()['parsed']['translated_text']

    def answer(self, query, metadata, first_request):
        if first_request:
            self.language = self.language_detection(query)
        else:
            query = self.translate(query)
        city = metadata.city.dict()
        # time to get the landmarks
        start = time.time()
        landmarks = self.get_nearby_landmarks(city)
        end = time.time()
        # time to get the response
        start = time.time()
        new_message = {
            "role": "user",
            "content": json.dumps(
                {
                    "current_city": city,
                    "near_by_landmarks": [landmark.dict() for landmark in landmarks],
                    "new_query": query,
                }
            ),
        }

        loc_info = self.is_location_information_seeking(query)

        additional_info = {}

        if loc_info["prediction"]:
            location = loc_info["location"]
            location_info = self.search_location(location)
            additional_info["location_info"] = {location: location_info}
            additional_info["location_info"]["wikipedia"] = self.get_wikipedia_article(
                location
            )
        else:
            additional_info["general_info"] = {
                "wikipedia": self.get_wikipedia_article(query)
            }
        new_system_prompt = {"role": "system", "content": self.system_prompt["content"]}
        new_system_prompt["content"] = new_system_prompt["content"].format(
            additional_info=json.dumps(additional_info), language=self.language
        )

        completion = self.model.chat_completion(
            messages=[new_system_prompt] + self.conversation + [new_message],
            response_format=CityWalkResponse
        )
        end = time.time()
        print("completion", completion)
        response = completion.choices[0].message.dict()["parsed"]
        new_response = {"role": "assistant", "content": response["speech"]}
        self.conversation.append(new_message)
        self.conversation.append(new_response)

        self.preferences = self.infer_user_preferences()
        print("Response:", response)  # 打印响应内容
        return CityWalkResponse(**response)  # 确保返回CityWalkResponse对象


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    agent = CityWalkAgent()
    landmarks = agent.get_nearby_landmarks(
        city={"latitude": 40.7128, "longitude": -74.0060}
    )
    print(landmarks)
    from pydantic import BaseModel

    class City(BaseModel):
        latitude: float
        longitude: float

    class MetaData(BaseModel):
        city: City
        is_first_request: bool

    city = City(latitude=40.7128, longitude=-74.0060)
    metadata = MetaData(city=city, is_first_request=True)

    response = agent.answer(
        "What are some interesting places to visit in New York?", metadata=metadata
    )
    print(response)

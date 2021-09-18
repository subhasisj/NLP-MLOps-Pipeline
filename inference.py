import os

import hydra
import torch
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

hydra.initialize("./configs")
cfg = hydra.compose(config_name="config.yaml")


class IntentDetectionModel:
    def __init__(self) -> None:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        self.model_path = os.path.join(cfg.model.model_dir, cfg.model.name)
        self.tokenizer_path = os.path.join(cfg.model.model_dir, cfg.model.tokenizer)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        ).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.pipeline = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer
        )

        self.id2label = {
            "0": "restaurant_reviews",
            "1": "nutrition_info",
            "2": "account_blocked",
            "3": "oil_change_how",
            "4": "time",
            "5": "weather",
            "6": "redeem_rewards",
            "7": "interest_rate",
            "8": "gas_type",
            "9": "accept_reservations",
            "10": "smart_home",
            "11": "user_name",
            "12": "report_lost_card",
            "13": "repeat",
            "14": "whisper_mode",
            "15": "what_are_your_hobbies",
            "16": "order",
            "17": "jump_start",
            "18": "schedule_meeting",
            "19": "meeting_schedule",
            "20": "freeze_account",
            "21": "what_song",
            "22": "meaning_of_life",
            "23": "restaurant_reservation",
            "24": "traffic",
            "25": "make_call",
            "26": "text",
            "27": "bill_balance",
            "28": "improve_credit_score",
            "29": "change_language",
            "30": "no",
            "31": "measurement_conversion",
            "32": "timer",
            "33": "flip_coin",
            "34": "do_you_have_pets",
            "35": "balance",
            "36": "tell_joke",
            "37": "last_maintenance",
            "38": "exchange_rate",
            "39": "uber",
            "40": "car_rental",
            "41": "credit_limit",
            "42": "oos",
            "43": "shopping_list",
            "44": "expiration_date",
            "45": "routing",
            "46": "meal_suggestion",
            "47": "tire_change",
            "48": "todo_list",
            "49": "card_declined",
            "50": "rewards_balance",
            "51": "change_accent",
            "52": "vaccines",
            "53": "reminder_update",
            "54": "food_last",
            "55": "change_ai_name",
            "56": "bill_due",
            "57": "who_do_you_work_for",
            "58": "share_location",
            "59": "international_visa",
            "60": "calendar",
            "61": "translate",
            "62": "carry_on",
            "63": "book_flight",
            "64": "insurance_change",
            "65": "todo_list_update",
            "66": "timezone",
            "67": "cancel_reservation",
            "68": "transactions",
            "69": "credit_score",
            "70": "report_fraud",
            "71": "spending_history",
            "72": "directions",
            "73": "spelling",
            "74": "insurance",
            "75": "what_is_your_name",
            "76": "reminder",
            "77": "where_are_you_from",
            "78": "distance",
            "79": "payday",
            "80": "flight_status",
            "81": "find_phone",
            "82": "greeting",
            "83": "alarm",
            "84": "order_status",
            "85": "confirm_reservation",
            "86": "cook_time",
            "87": "damaged_card",
            "88": "reset_settings",
            "89": "pin_change",
            "90": "replacement_card_duration",
            "91": "new_card",
            "92": "roll_dice",
            "93": "income",
            "94": "taxes",
            "95": "date",
            "96": "who_made_you",
            "97": "pto_request",
            "98": "tire_pressure",
            "99": "how_old_are_you",
            "100": "rollover_401k",
            "101": "pto_request_status",
            "102": "how_busy",
            "103": "application_status",
            "104": "recipe",
            "105": "calendar_update",
            "106": "play_music",
            "107": "yes",
            "108": "direct_deposit",
            "109": "credit_limit_change",
            "110": "gas",
            "111": "pay_bill",
            "112": "ingredients_list",
            "113": "lost_luggage",
            "114": "goodbye",
            "115": "what_can_i_ask_you",
            "116": "book_hotel",
            "117": "are_you_a_bot",
            "118": "next_song",
            "119": "change_speed",
            "120": "plug_type",
            "121": "maybe",
            "122": "w2",
            "123": "oil_change_when",
            "124": "thank_you",
            "125": "shopping_list_update",
            "126": "pto_balance",
            "127": "order_checks",
            "128": "travel_alert",
            "129": "fun_fact",
            "130": "sync_device",
            "131": "schedule_maintenance",
            "132": "apr",
            "133": "transfer",
            "134": "ingredient_substitution",
            "135": "calories",
            "136": "current_location",
            "137": "international_fees",
            "138": "calculator",
            "139": "definition",
            "140": "next_holiday",
            "141": "update_playlist",
            "142": "mpg",
            "143": "min_payment",
            "144": "change_user_name",
            "145": "restaurant_suggestion",
            "146": "travel_notification",
            "147": "cancel",
            "148": "pto_used",
            "149": "travel_suggestion",
            "150": "change_volume",
        }

    def predict_intent(self, text):

        pred = model.pipeline("I want to book a cab to the airport")[0]["label"].split(
            "_"
        )[1]
        return self.id2label[pred]


if __name__ == "__main__":
    model = IntentDetectionModel()
    pred = model.pipeline("I want to book a cab to the airport")[0]["label"].split("_")[
        1
    ]
    print(model.id2label[pred])


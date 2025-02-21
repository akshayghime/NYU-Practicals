

def send_coupon(user):
    if user.is_new:
        if user.device == "ios":
            return True
        elif user.device == "android" and user.os_major_version >= 11:
            return True
        
    if (
        user.referral_source in ("facebook","organic")
        and user.days_since_last_visit > 5
    ):
        return True
    
    return False


user = [User(is_new=False, device="ios",os_major_version = 13, referral_source= "facebook", days_since_last_visit = 10), 
        User(is_new=False, device="android",os_major_version = 11, referral_source= "facebook", days_since_last_visit = None)]



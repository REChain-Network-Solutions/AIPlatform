<?php

namespace App\Extensions\Chatbot\System\Helpers;

use App\Helpers\Classes\MarketplaceHelper;

class ChatbotHelper
{
    public static function existChannels(): bool
    {
        return MarketplaceHelper::isRegistered('chatbot-telegram') || MarketplaceHelper::isRegistered('chatbot-whatsapp');
    }

    public static function channels()
    {
        //		return [
        //
        //		]
    }
}

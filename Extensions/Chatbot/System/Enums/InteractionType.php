<?php

namespace App\Extensions\Chatbot\System\Enums;

enum InteractionType: string
{
    case AUTOMATIC_RESPONSE = 'automatic_response'; // Automatic Response (AI)

    case HUMAN_SUPPORT = 'human_support'; // Human Support (Agent)

    case SMART_SWITCH = 'smart_switch'; // Smart Switch (AI or Agent)
}

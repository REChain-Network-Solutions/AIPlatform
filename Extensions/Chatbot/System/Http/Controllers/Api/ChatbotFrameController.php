<?php

namespace App\Extensions\Chatbot\System\Http\Controllers\Api;

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cookie;

class ChatbotFrameController extends Controller
{
    public function frame(Request $request, Chatbot $chatbot): View
    {
        $session = $this->getVisitor();

        $conversations = ChatbotConversation::query()
            ->where('chatbot_id', $chatbot->getAttribute('id'))
            ->where('session_id', $session)
            ->get();

        return view('chatbot::frame', compact('chatbot', 'session', 'conversations'));
    }

    protected function getVisitor(): string
    {
        $cookie = Cookie::has('CHATBOT_VISITOR');

        if ($cookie) {
            return Cookie::get('CHATBOT_VISITOR');
        }

        $sessionId = md5(uniqid(mt_rand(), true));

        Cookie::queue('CHATBOT_VISITOR', $sessionId, 60 * 24 * 365);

        return $sessionId;
    }
}

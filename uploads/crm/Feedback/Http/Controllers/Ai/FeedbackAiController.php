<?php

namespace Modules\Feedback\Http\Controllers\Ai;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Modules\Feedback\AI\OpenAIClient;

class FeedbackAiController extends Controller
{
    public function generate(Request $request)
    {
        $client = new OpenAIClient();
        $data = $client->generateFeedback([
            'prompt' => $request->string('prompt', ''),
        ]);
        return response()->json($data);
    }

    public function suggestReply(Request $request)
    {
        $client = new OpenAIClient();
        $data = $client->suggestReply([
            'text' => $request->string('feedback', $request->string('text','')),
        ]);
        return response()->json($data);
    }

    public function smoke()
    {
        $client = new OpenAIClient();
        return response()->json($client->smoke());
    }
}

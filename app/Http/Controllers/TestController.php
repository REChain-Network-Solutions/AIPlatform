<?php

namespace App\Http\Controllers;

use App\Helpers\Classes\ApiHelper;
use Illuminate\Http\Request;

class TestController extends Controller
{
    public function test(Request $request)
    {
        return view('test');
    }

    public function stream(Request $request, string $model)
    {
        // Disable PHP buffering
        @ini_set('output_buffering', 'off');
        @ini_set('zlib.output_compression', false);
        @ini_set('implicit_flush', true);
        ob_implicit_flush(true);
        while (ob_get_level() > 0) {
            @ob_end_flush();
        }

        return response()->stream(function () use ($model) {
            $output = '';
            $responsedText = '';
            $history = [];
            $total_used_tokens = 0;

            // --- OpenAI GPT models ---
            if (str_starts_with($model, 'gpt')) {
                ApiHelper::setOpenAiKey();
                $history[] = ['role' => 'user', 'content' => 'Hello, write essay about cats.'];

                $stream = \OpenAI\Laravel\Facades\OpenAI::responses()->createStreamed([
                    'model'             => $model,
                    'input'             => $history,
                    'max_output_tokens' => 2000,
                    'temperature'       => 1.0,
                    'stream'            => true,
                ]);

                foreach ($stream as $response) {
                    if (connection_aborted()) {
                        return;
                    }

                    if ($response->event === 'response.output_text.delta' && isset($response->response->delta)) {
                        $text = $response->response->delta;
                        $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $text);
                        $output .= $messageFix;
                        $responsedText .= $text;
                        $total_used_tokens += countWords($text);

                        echo PHP_EOL;
                        // echo "event: data\n";
                        echo "data: {$messageFix}";
                        echo "\n\n";
                        @ob_flush();
                        @flush();
                    }
                }

                echo "data: [DONE]\n\n";

                return;
            }

            // --- Gemini placeholder ---
            if ($model === 'test') {
                for ($i = 1; $i <= 5; $i++) {
                    if (connection_aborted()) {
                        return;
                    }

                    echo "data: GEMINI chunk {$i}\n\n";
                    @ob_flush();
                    @flush();
                    sleep(1);
                }
                echo "data: [DONE]\n\n";

                return;
            }

        }, 200, [
            'Content-Type'      => 'text/event-stream',
            'Cache-Control'     => 'no-cache, must-revalidate',
            'Connection'        => 'keep-alive',
            'X-Accel-Buffering' => 'no', // disable nginx buffering
        ]);
    }
}

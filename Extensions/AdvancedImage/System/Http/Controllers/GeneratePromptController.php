<?php

namespace App\Extensions\AdvancedImage\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use OpenAI\Laravel\Facades\OpenAI;

class GeneratePromptController extends Controller
{
    public function __invoke(Request $request)
    {
        Helper::setOpenAiKey();

        $request->validate(['image' => 'required|image']);

        $image = $request->file('image')->store('image-editor', ['disk' => 'public']);

        $driver = Entity::driver(EntityEnum::GPT_4_TURBO);

        try {
            $driver->redirectIfNoCreditBalance();
        } catch (Exception $e) {
            return response()->json([
                'prompt'  => __($e->getMessage()),
                'status'  => 'error',
            ]);
        }

        $completion = OpenAI::chat()->create([
            'model'    => 'gpt-4-turbo',
            'messages' => [
                [
                    'role'    => 'system',
                    'content' => 'Analyze the image and generate a detailed, high-quality prompt suitable for recreating the scene using an image generation model. Output only the prompt, with no additional text or labels.',
                ],
                [
                    'role'    => 'user',
                    'content' => [
                        [
                            'type'      => 'image_url',
                            'image_url' => [
                                'url' => Storage::disk('uploads')->url(str_replace('uploads', '', $image)),
                            ],
                        ],
                    ],
                ],
            ],
        ]);

        if (isset($completion['choices'][0]['message']['content'])) {

            $driver->input($completion['choices'][0]['message']['content'])->calculateCredit()->decreaseCredit();

            $data = [
                'status' => 'success',
                'prompt' => $completion['choices'][0]['message']['content'],
            ];
        } else {
            $data = [
                'status' => 'error',
                'prompt' => 'No response from ai',
            ];
        }

        return response()->json($data);
    }
}

<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\Company;
use App\Models\Product;
use App\Models\Setting;
use Exception;
use Illuminate\Http\Request;
use OpenAI\Laravel\Facades\OpenAI;

class GenerateContentController extends Controller
{
    public ?Setting $setting;

    public function __construct()
    {
        $this->setting = Setting::getCache();

        Helper::setOpenAiKey();
    }

    public function generateContent(Request $request)
    {
        $driver = Entity::driver();
        $driver->redirectIfNoCreditBalance();

        try {
            $platform = $request->platform;
            $company_id = $request->company_id;

            $company = Company::find($company_id);
            $camp_target = $request->camp_target;

            $productIds = $request->productIds;
            $products = Product::whereIn('id', $productIds)->get();
            $productDescriptions = $products->map(function ($product, $index) {
                switch ($product->type) {
                    case 0: // Product
                        return '(product' . ($index + 1) . ': ' . $product->name . ')';
                    case 1: // Service
                        return '(service' . ($index + 1) . ': ' . $product->name . ')';
                    default: // Other
                        return '(' . ($index + 1) . ': ' . $product->name . ')';
                }
            });
            $productString = $productDescriptions->implode(', ');

            $driver
                ->input($productString)
                ->calculateCredit()->decreaseCredit();

            $tone = $request->tone;
            $seo = $request->seo;
            $topics = $request->topics;
            $num_res = $request->num_res;
            $cam_injected_name = $request->cam_injected_name;

            $topics = is_array($topics) ? implode(', ', $topics) : $topics;

            $prompt = "Craft a with a maximum length of $num_res characters, without any emojis, emoticons a text compelling social media post for $platform platform, to promote a " . ($seo ? 'SEO optimized. ' : '') . 'campaign ' . ($company->name ? 'by ' . $company->name . '. ' : '. ') . ($camp_target ? 'The campaign aims to reach: [' . $camp_target . ']. ' : '') . 'Focus on this provided: [' . $productString . '] ' . ($cam_injected_name ? 'Campaign Name: ' . $cam_injected_name . '. ' : '') . ($topics ? 'Topics: ' . $topics . '. ' : '') . ($seo ? 'SEO optimized. ' : '') . ($tone ? 'Tone of Voice: ' . $tone . '. ' : '') . 'Do not include or links' . ($company->website ? 'other than the website.' : '.') . '. Must not ever increase the length of the post.';

            $driver = Entity::driver(EntityEnum::tryFrom($this->setting->openai_default_model));

            $driver->redirectIfNoCreditBalance();

            $completion = OpenAI::chat()->create([
                'model'    => $this->setting->openai_default_model,
                'messages' => [[
                    'role'    => 'user',
                    'content' => $prompt,
                ]],
            ]);

            $response = $completion['choices'][0]['message']['content'];

            $driver->input($response)->calculateCredit()->decreaseCredit();

            return response()->json([
                'result' => $completion['choices'][0]['message']['content'],
            ]);
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }
}

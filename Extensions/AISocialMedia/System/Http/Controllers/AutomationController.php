<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers;

use App\Domains\Entity\EntityStats;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\AISocialMedia\System\Models\AutomationCampaign;
use App\Extensions\AISocialMedia\System\Models\ScheduledPost;
use App\Extensions\AISocialMedia\System\Services\AutomationService;
use App\Helpers\Classes\ApiHelper;
use App\Http\Controllers\Controller;
use App\Models\Company;
use App\Models\Product;
use App\Models\Setting;
use App\Models\SettingTwo;
use Exception;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use OpenAI\Laravel\Facades\OpenAI;
use Throwable;

class AutomationController extends Controller
{
    protected $settings;

    protected $settings_two;

    public function __construct(public AutomationService $automationService)
    {
        $this->setting = Setting::getCache();
        $this->setting_two = SettingTwo::getCache();
        ApiHelper::setOpenAiKey();
    }

    // automation
    public function index()
    {
        $apiKey = ApiHelper::setOpenAiKey();

        $len = strlen($apiKey);
        $len = max($len, 6);
        $parts[] = substr($apiKey, 0, $l[] = rand(1, $len - 5));
        $parts[] = substr($apiKey, $l[0], $l[] = rand(1, $len - $l[0] - 3));
        $parts[] = substr($apiKey, array_sum($l));

        $apikeyPart1 = base64_encode($parts[0]);
        $apikeyPart2 = base64_encode($parts[1]);
        $apikeyPart3 = base64_encode($parts[2]);
        $apiUrl = base64_encode('https://api.openai.com/v1/chat/completions');

        $step = 1;

        $platforms = AutomationService::platforms();

        return view('ai-social-media::index', compact(
            'step',
            'apikeyPart1',
            'apikeyPart2',
            'apikeyPart3',
            'apiUrl',
            'platforms'
        ));
    }

    public function nextStep(Request $request)
    {
        $step = $request?->step;
        $platform_id = $request?->platform_id;
        $company_id = $request?->company_id;
        $product_id = $request->input('product_id', []);
        $camp_id = $request?->camp_id;
        $camp_target = $request?->camp_target;
        $topics = $request->input('topics', []);

        $tone = $request?->tone;
        $num_res = $request?->num_res;
        $vis_format = $request?->vis_format;
        $vis_ratio = $request?->vis_ratio;
        $time = $request?->time;
        $repeat_period = $request?->repeat_period;

        $sendMail = ($request?->sendMail == 'on' || $request?->sendMail) ? true : false;
        $review = ($request?->review == 'on' || $request?->review) ? true : false;
        $seo = ($request?->seo == 'on' || $request?->seo) ? true : false;
        $is_img = ($request?->is_img == 'on' || $request?->is_img) ? true : false;
        $repeat = ($request?->repeat == 'on' || $request?->repeat) ? true : false;

        $cam_injected_name = $request?->cam_injected_name;

        $date = $request?->date;

        if ($step == 7) {
            $this->lastStep(
                $review,
                $date,
                $cam_injected_name,
                $time,
                $repeat,
                $repeat_period,
                $platform_id,
                $company_id,
                $product_id,
                $camp_id,
                $camp_target,
                $topics,
                $seo,
                $is_img,
                $tone,
                $num_res,
                $vis_format,
                $vis_ratio,
                $sendMail
            );

            return redirect()->route('dashboard.user.automation.list')->with(['message' => __('Post Task Saved Successfuly'), 'type' => 'success']);
        }

        [$apikeyPart1, $apikeyPart2, $apikeyPart3, $apiUrl] = $this->appKeys();

        return view('ai-social-media::index', compact('step', 'date', 'cam_injected_name', 'platform_id', 'company_id', 'product_id', 'camp_id', 'camp_target', 'topics', 'seo', 'is_img', 'tone', 'num_res', 'vis_format', 'vis_ratio', 'sendMail',
            'apikeyPart1',
            'apikeyPart2',
            'apikeyPart3',
            'apiUrl'));
    }

    private function appKeys(): array
    {
        $apiKey = ApiHelper::setOpenAiKey();
        $len = strlen($apiKey);
        $len = max($len, 6);
        $parts[] = substr($apiKey, 0, $l[] = rand(1, $len - 5));
        $parts[] = substr($apiKey, $l[0], $l[] = rand(1, $len - $l[0] - 3));
        $parts[] = substr($apiKey, array_sum($l));

        return [
            base64_encode($parts[0]),
            base64_encode($parts[1]),
            base64_encode($parts[2]),
            base64_encode('https://api.openai.com/v1/chat/completions'),
        ];
    }

    private function lastStep($review, $date, $cam_injected_name, $time, $repeat, $repeat_period, $platform_id, $company_id, $product_id, $camp_id, $camp_target, $topics, $seo, $is_img, $tone, $num_res, $vis_format, $vis_ratio, $sendMail): void
    {
        $automationPlatform = $this->automationService->findAutomationPlatform($platform_id);

        try {
            $platform = 'Any';
            switch ($platform_id) {
                case '1':
                    $platform = 'Twitter.com/X.com';

                    break;
                case '2':
                    $platform = 'Linkedin.com';

                    break;
                default:
                    $platform = 'Any';

                    break;
            }
            $company = Company::find($company_id);
            $products = Product::whereIn('id', $product_id)->get();
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
            $topics = is_array($topics) ? implode(', ', $topics) : $topics;
            $camp_target = str_replace("\r\n", ', ', $camp_target);

            $prompt = "Craft a with a maximum length of $num_res characters, without any emojis, emoticons a text compelling social media post for $platform platform, to promote a " . ($seo ? 'SEO optimized. ' : '') . 'campaign ' . ($company->name ? 'by ' . $company->name . '. ' : '. ') . ($camp_target ? 'The campaign aims to reach: [' . $camp_target . ']. ' : '') . 'Focus on this provided: ' . $productString . '] ' . ($cam_injected_name ? 'Campaign Name: ' . $cam_injected_name . '. ' : '') . ($topics ? 'Topics: ' . $topics . '. ' : '') . ($tone ? 'Tone of Voice: ' . $tone . '. ' : '') . 'Do not include or links' . ($company->website ? 'other than the website.' : '.') . '. Must not ever increase the length of the post.';

            $schedule = new ScheduledPost;
            $schedule->automation_platform_id = $automationPlatform?->id;
            $schedule->user_id = auth()->user()->id;
            $schedule->company_id = $company_id;
            $schedule->platform = $platform_id;
            $schedule->products = $productString;
            $schedule->campaign_name = $cam_injected_name;
            $schedule->campaign_target = $camp_target;
            $schedule->topics = $topics;
            $schedule->is_seo = $seo;
            $schedule->tone = $tone;
            $schedule->length = $num_res;
            $schedule->is_email = $sendMail;
            $schedule->is_repeated = $repeat;
            $schedule->repeat_period = $repeat_period;
            $schedule->repeat_start_date = date('Y-m-d', strtotime($date));
            $schedule->repeat_time = $time;
            $schedule->visual_format = $vis_format;
            $schedule->visual_ratio = $vis_ratio;
            $schedule->prompt = $prompt;
            $schedule->last_run_date = now()->addDays(-1);
            $schedule->save();
        } catch (Throwable $th) {
            // TODO: Log the error..
        }
    }

    // company
    public function getProducts($company_id): JsonResponse
    {
        $products = Product::where('company_id', $company_id)->where('user_id', auth()->user()->id)->get();

        return response()->json($products);
    }

    // campaign
    public function getCampaignTarget($campaign_id): JsonResponse
    {
        $campaign = AutomationCampaign::query()->where('id', $campaign_id)->firstOrFail();

        return response()->json($campaign->target_audience ?? '');
    }

    public function campaignAddOrUpdate($id = null)
    {
        if ($id == null) {
            $item = null;
        } else {
            $item = AutomationCampaign::where('id', $id)->where('user_id', auth()->user()->id)->firstOrFail();
        }

        // TODO: eski extension dosyalarinda boyle bir blade goremedim
        return view('ai-social-media::campaigns.form', compact('item'));
    }

    public function campaignList()
    {
        $list = AutomationCampaign::orderBy('name', 'asc')->get();

        return view('ai-social-media::campaigns.list', compact('list'));
    }

    public function campaignDelete($id = null)
    {
        $item = AutomationCampaign::where('id', $id)->firstOrFail();
        $item->delete();

        return back()->with(['message' => 'Deleted Successfully', 'type' => 'success']);
    }

    public function campaignAddOrUpdateSave(Request $request)
    {

        if ($request->cam_id == 'undefined' || $request->cam_id == null) {
            $cam = new AutomationCampaign;
        } else {
            $cam = AutomationCampaign::where('id', $request->cam_id)->firstOrFail();
        }

        try {
            $cam->name = $request->cam_name;
            $cam->target_audience = $request->cam_target;
            $cam->user_id = auth()->user()->id;
            $cam->save();

            return response()->json(true);
        } catch (Throwable $th) {
            return response()->json(false);
        }

    }

    public function generateCampaignContent(Request $request)
    {

        $driver = Entity::driver(EntityEnum::tryFrom($this->setting->openai_default_model));
        $driver->redirectIfNoCreditBalance();

        try {
            $campaign_name = $request->campaign_name ?? 'any';
            // $productIds = $repeat->productIds?? 0; later we can add products to improve the output..
            $completion = OpenAI::chat()->create([
                'model'    => $this->setting->openai_default_model,
                'messages' => [[
                    'role'    => 'user',
                    'content' => "Generate only a list of target audience attributes, including demographics, interests, and pain points, for the purpose of $campaign_name campaign. Must result as array json data only. Result format is [attribute1, attribute2, ..., attributen]. Ensure that the result does not contain backticks (\`) or the string \"```json\".",
                ]],
            ]);
            $response = $completion['choices'][0]['message']['content'];
            $driver->input($response)->calculateCredit()->decreaseCredit();

            return response()->json(['result' => $completion['choices'][0]['message']['content']]);
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    public function generateCampaignTopics(Request $request)
    {
        $driver = Entity::driver(EntityEnum::tryFrom($this->setting->openai_default_model));
        $driver->redirectIfNoCreditBalance();

        try {
            $campaign_name = $request->campaign_name ?? 'any';

            $completion = OpenAI::chat()->create([
                'model'    => $this->setting->openai_default_model,
                'messages' => [[
                    'role'    => 'user',
                    'content' => "Generate only a list of max 10 topics, of $campaign_name campaign. Must resut as array json data. Result format is [topic1, topic2, ..., topicn]. Ensure that the result does not contain backticks (\`) or the string \"```json\".",
                ]],
            ]);

            $response = $completion['choices'][0]['message']['content'];

            $driver->input($response)->calculateCredit()->decreaseCredit();

            return response()->json(['result' => $completion['choices'][0]['message']['content']]);
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    public function updateAutomation(Request $request)
    {
        $user = Auth::user();

        try {
            $data = $request->getContent();
            $decodedData = json_decode($data);
            if ($decodedData->type === 'TOKENS') {
                $driver = Entity::driver(EntityEnum::tryFrom($this->settings->openai_default_model));
                $driver->setCalculatedInputCredit(2);
                $driver->input($data)->calculateCredit()->decreaseCredit();
            }
            $wordModelsCredits = EntityStats::word()->forUser($user)->totalCredits();
            $imageModelsCredits = EntityStats::image()->forUser($user)->totalCredits();

            return response()->json(['result' => 'success', 'remain_words' => (string) $wordModelsCredits, 'remain_images' => (string) $imageModelsCredits]);
        } catch (Exception $e) {
            return response()->json([
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    public function getCompany(Request $request)
    {
        $company_id = $request->id;
        $company = Company::find($company_id);

        return response()->json($company);
    }

    public function getSelectedProducts(Request $request)
    {
        $ids_array = $request->ids_array;
        $selectedProducts = Product::whereIn('id', $ids_array)->get();

        return response()->json($selectedProducts);
    }

    public function scheduledPosts()
    {
        return view('ai-social-media::list');
    }

    public function scheduledPostsDelete($id): RedirectResponse
    {
        $item = ScheduledPost::where('id', $id)->firstOrFail();
        $item->delete();

        return back()->with(['message' => 'Deleted Successfully', 'type' => 'success']);
    }

    public function scheduledPostsEdit(Request $request, $id): RedirectResponse
    {
        $item = ScheduledPost::query()->where('id', $id)->firstOrFail();
        if ($item) {
            $item->is_repeated = ($request?->repeat == 'on' || $request?->repeat) ? true : false;
            $item->repeat_period = $request->repeat_period;
            $item->repeat_start_date = date('Y-m-d', strtotime($request?->date));
            $item->repeat_time = $request?->time;
            $item->save();
        }

        return back()->with([
            'message' => 'Updated Successfully',
            'type'    => 'success',
        ]);
    }
}

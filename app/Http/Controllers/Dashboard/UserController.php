<?php

namespace App\Http\Controllers\Dashboard;

use App\Actions\CreateActivity;
use App\Console\Commands\FluxProQueueCheck;
use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Models\Entity;
use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use App\Enums\AiImageStatusEnum;
use App\Extensions\AiChatProImageChat\System\Models\AiChatProImageModel;
use App\Extensions\AIImagePro\System\Models\AiImageProModel;
use App\Extensions\AiVideoPro\System\Models\UserFall;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Helpers\Classes\MarketplaceHelper;
use App\Http\Controllers\Controller;
use App\Http\Controllers\Finance\PaymentProcessController;
use App\Jobs\SendInviteEmail;
use App\Models\AccountDeletionReqs;
use App\Models\Currency;
use App\Models\Folders;
use App\Models\Integration\Integration;
use App\Models\Integration\UserIntegration;
use App\Models\OpenAIGenerator;
use App\Models\OpenaiGeneratorChatCategory;
use App\Models\OpenaiGeneratorFilter;
use App\Models\Setting;
use App\Models\SettingTwo;
use App\Models\Team\Team;
use App\Models\User;
use App\Models\UserAffiliate;
use App\Models\UserDocsFavorite;
use App\Models\UserFavorite;
use App\Models\UserOpenai;
use App\Models\UserOpenaiChat;
use App\Models\UserOrder;
use App\Models\Voice\ElevenlabVoice;
use App\Services\Dashboard\UserDashboardService;
use App\Services\ElevenlabsService;
use App\Services\GatewaySelector;
use App\Services\Orders\OrdersExportService;
use enshrined\svgSanitize\Sanitizer;
use Exception;
use Illuminate\Contracts\Auth\Authenticatable;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Pagination\LengthAwarePaginator;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Illuminate\Validation\Rules\Password;
use MagicAI\Updater\Facades\Updater;
use Throwable;

class UserController extends Controller
{
    public function __construct(
        public UserDashboardService $service,
    ) {}

    public function checkStatus(Request $request)
    {
        $data = UserOpenai::query()
            ->where('status', 'IN_QUEUE')
            ->where('response', 'FL')
            ->get();

        if ($data->isEmpty()) {
            return response()->json([]);
        }

        FluxProQueueCheck::updateFluxProImages();

        return [
            'data' => UserOpenai::query()
                ->whereIn('id', $data->pluck('id')->toArray())
                ->where('status', '<>', 'IN_QUEUE')
                ->get()
                ->map(function ($item) {
                    $item->setAttribute('imgId', 'img-' . $item->response . '-' . $item->id);
                    $item->setAttribute('payloadId', 'img-' . $item->response . '-' . $item->id . '-payload');
                    $item->setAttribute('img', ThumbImage($item->output));

                    return $item;
                }),

        ];
    }

    public function redirect(Request $request)
    {
        $route = 'dashboard.user.index';

        if ($request->user()->isAdmin() && Helper::appIsNotDemo()) {
            $route = 'dashboard.admin.index';
        }

        return to_route($route);
    }

    public function index()
    {
        $user = Auth::user();
        $user->load('folders');
        $this->service->setCache();
        $ongoingPayments = null;
        // $ongoingPayments = self::prepareOngoingPaymentsWarning();
        if (Helper::appIsNotDemo()) {
            PaymentProcessController::checkUnmatchingSubscriptions();
        }

        return view('panel.user.dashboard', [
            'user'              => $user,
            'team'              => $this->getTeam($user),
            'ongoingPayments'   => $ongoingPayments,
            'recently_launched' => UserOpenai::query()
                ->where('user_id', $user->id)
                ->orderBy('updated_at', 'desc')
                ->limit(5)
                ->get(),
        ]);
    }

    public function markTourSeen()
    {
        $user = Auth::user();
        if ($user) {
            $user->tour_seen = true;
            $user->save();
        }

        return response()->json(['status' => 'success']);
    }

    public function getTeam(User|Authenticatable|null $user)
    {
        if ($team = $user?->myCreatedTeam) {

            if ($team->allow_seats != $user?->relationPlan?->plan_allow_seat) {
                $team->allow_seats = $user?->relationPlan?->plan_allow_seat ?: 0;
                $team->save();
            }

            return $team;
        }

        $allow_seats = $user?->isAdmin() ? 100 : $user?->relationPlan?->plan_allow_seat;

        return Team::query()->firstOrCreate([
            'user_id' => auth()->id(),
        ], [
            'name'        => $user?->fullName(),
            'allow_seats' => $allow_seats ?: 0,
        ]);
    }

    public function prepareOngoingPaymentsWarning()
    {
        $ongoingPayments = PaymentProcessController::checkForOngoingPayments();

        if ($ongoingPayments) {
            return $ongoingPayments;
        }

        return null;
    }

    public function openAIList()
    {
        abort_if(Helper::setting('feature_ai_writer') == 0, 404);

        return view('panel.user.openai.list', [
            'list' => OpenAIGenerator::query()
                ->where(function ($query) {
                    $query->where('user_id', Auth::id())
                        ->orWhereNull('user_id');
                })->where('active', true)->get(),
            'filters' => OpenaiGeneratorFilter::query()->where(function ($query) {
                $query->where('user_id', auth()->user()->id)
                    ->orWhereNull('user_id');
            })->orderBy('name', 'desc')->get(),
        ]);
    }

    public function openAICustomList()
    {
        abort_if(setting('user_ai_writer_custom_templates', 0) == 0, 404);

        $list = OpenAIGenerator::query()
            ->with('user')
            ->orderBy('title', 'asc')
            ->where('custom_template', 1)
            ->where('user_id', Auth::id())
            ->get();

        return view('panel.user.openai.custom.list', compact('list'));
    }

    public function openAICustomAddOrUpdateSave(Request $request)
    {
        $userId = Auth::id();
        if ($request->template_id != 'undefined') {
            $template = OpenAIGenerator::where('id', $request->template_id)->where('user_id', $userId)->firstOrFail();
        } else {
            $template = new OpenAIGenerator;
        }

        // Set basic template attributes
        $template->title = $request->title;
        $template->description = $request->description;
        $template->image = $request->image;
        $template->color = $request->color;
        $template->prompt = $request->prompt;
        $template->filters = __('My Templates');
        $template->premium = 0;
        $template->active = 1;
        $template->slug = Str::slug($request->title) . '-' . Str::random(6);
        $template->type = 'text';
        $template->custom_template = 1;
        $template->user_id = $userId;

        // Process input data by type
        $inputDataByType = json_decode($request->input_data_by_type, true);

        $allquestions = [];
        foreach ($inputDataByType as $inputType => $inputs) {
            foreach ($inputs as $input) {
                // Save input data as arrays
                $inputArray = [
                    'name'        => Str::slug($input['inputName']),
                    'question'    => $input['inputName'],
                    'description' => $input['inputDescription'],
                    'type'        => $inputType,
                ];
                // If input type is select, include select list values
                if ($inputType === 'select') {
                    $inputArray['selectListValues'] = $input['selectListValues'];
                }

                // Save input data array into questions array
                $allquestions[] = $inputArray;
            }
        }
        $questions = json_encode($allquestions, JSON_UNESCAPED_SLASHES);
        $template->questions = $questions;

        $template->save();

        if (OpenaiGeneratorFilter::where('name', __('My Templates'))->first() == null) {
            $newFilter = new OpenaiGeneratorFilter;
            $newFilter->name = __('My Templates');
            $newFilter->save();
        }

        $setting = Setting::getCache();
        $freeOpenAiItems = $setting->free_open_ai_items;
        $freeOpenAiItems[] = $template->slug;
        $setting->update([
            'free_open_ai_items' => $freeOpenAiItems ?: [],
        ]);
    }

    public function openAICustomAddOrUpdate($id = null)
    {
        $userId = Auth::id();
        if ($id == null) {
            $template = null;
        } else {
            $template = OpenAIGenerator::where('id', $id)->where('user_id', $userId)->firstOrFail();
        }
        $filters = OpenaiGeneratorFilter::orderBy('name', 'desc')->get();

        return view('panel.user.openai.custom.form', compact('template', 'filters'));
    }

    public function openAICustomDelete($id = null)
    {
        $userId = Auth::id();
        $template = OpenAIGenerator::where('id', $id)->where('user_id', $userId)->firstOrFail();
        $template->delete();

        return back()->with(['message' => __('Deleted Successfully'), 'type' => 'success']);
    }

    public function openAIFavoritesList()
    {
        return view('panel.user.openai.list_favorites');
    }

    // docsFavorite
    public function docsFavorite(Request $request)
    {
        $documentId = $request->id;
        $externalDocument = $this->parseExternalDocumentIdentifier((string) $request->id);

        if ($externalDocument !== null) {
            $documentId = $this->resolveExternalDocumentProxyId($externalDocument);

            if ($documentId === null) {
                return response()->json(['message' => __('Document not found.')], 404);
            }
        }

        $exists = isFavoritedDoc($documentId);
        if ($exists) {
            $favorite = UserDocsFavorite::where('user_openai_id', $documentId)->where('user_id', Auth::id())->firstOrFail();
            $favorite->delete();
            $action = 'unfavorite';
        } else {
            $favorite = new UserDocsFavorite;
            $favorite->user_id = Auth::id();
            $favorite->user_openai_id = $documentId;
            $favorite->save();
            $action = 'favorite';
        }

        return response()->json(['action' => $action]);
    }

    public function openAIFavorite(Request $request)
    {
        $exists = isFavorited($request->id);
        if ($exists) {
            $favorite = UserFavorite::where('openai_id', $request->id)->where('user_id', Auth::id())->firstOrFail();
            $favorite->delete();
            $action = 'unfavorite';
        } else {
            $favorite = new UserFavorite;
            $favorite->user_id = Auth::id();
            $favorite->openai_id = $request->id;
            $favorite->save();
            $action = 'favorite';
        }

        return response()->json(['action' => $action]);
    }

    public function openAIGenerator(Request $request, $slug)
    {
        if (MarketplaceHelper::isRegistered('ai-image-pro') && setting('ai_image_pro_display_type', 'both_fm') === 'ai_image') {
            return redirect()->route('dashboard.user.ai-image-pro.index');
        }

        $openai = OpenAIGenerator::whereSlug($slug)->firstOrFail();
        if ($slug === 'ai_image_generator') {
            // FluxProQueueCheck::updateFluxProImages();
        }

        if ($slug === 'ai_video_to_video' && ! MarketplaceHelper::isRegistered('ai-video-to-video')) {
            return redirect()->route('dashboard.user.index')->with(['message' => __('Feature not available'), 'type' => 'error']);
        }

        $userOpenai = $this->openai($request, null)
            ->where('openai_id', $openai->id)
            ->when($slug === 'ai_image_generator' && Schema::hasColumn('user_openai', 'is_fashion_studio'), function ($query) {
                $query->where('is_fashion_studio', false);
            })
            ->orderBy('created_at', 'desc')
            ->paginate(5);

        $elevenlabServiceVoice = [];
        $elevenlabs = null;
        $matchedVoices = [];

        if ($openai->type === 'voiceover' || $openai->type === 'isolator') {
            $elevenlabs = ElevenlabVoice::query()
                ->select('voice_id', 'name')
                ->where('status', 1)
                ->where('user_id', Auth::id())
                ->whereNotNull('voice_id')
                ->get();

            $service = new ElevenlabsService;
            if (! is_array($service->getVoices()) && ! empty($service->getVoices())) {
                $elevenlabServiceVoice = json_decode($service->getVoices(), true);
            }

            $userVoiceIds = $elevenlabs->pluck('voice_id')->toArray();
            foreach ($elevenlabServiceVoice as $key => $voice) {
                if (! in_array($voice['voice_id'], $userVoiceIds) && $voice['category'] != 'premade') {
                    unset($elevenlabServiceVoice[$key]);
                }
            }
        }

        $list = OpenAIGenerator::query()
            ->where(function ($query) {
                $query->where('user_id', Auth::id())
                    ->orWhereNull('user_id');
            })->where('active', true)->get(['id', 'title', 'slug', 'image']);

        return view(
            'panel.user.openai.generator',
            compact('openai', 'list', 'userOpenai', 'elevenlabs', 'elevenlabServiceVoice')
        );
    }

    public function openAIGeneratorWorkbook($slug)
    {
        $openai = OpenAIGenerator::whereSlug($slug)->firstOrFail();
        $settings2 = SettingTwo::getCache();
        $apiUrl = base64_encode('https://api.openai.com/v1/chat/completions');

        if (setting('default_ai_engine', EngineEnum::OPEN_AI->value) == EngineEnum::ANTHROPIC->value) {
            $apiUrl = base64_encode('https://api.anthropic.com/v1/messages');
        }

        if ($settings2->openai_default_stream_server == 'backend') {
            $apikeyPart1 = base64_encode(random_int(1, 100));
            $apikeyPart2 = base64_encode(random_int(1, 100));
            $apikeyPart3 = base64_encode(random_int(1, 100));
        } else {
            // Fetch the Site Settings object with openai_api_secret
            $apiKey = ApiHelper::setOpenAiKey();

            if (setting('default_ai_engine', EngineEnum::OPEN_AI->value) == EngineEnum::ANTHROPIC->value) {
                $apiKey = ApiHelper::setAnthropicKey();
            }

            $len = strlen($apiKey);
            $len = max($len, 6);
            $parts[] = substr($apiKey, 0, $l[] = random_int(1, $len - 5));
            $parts[] = substr($apiKey, $l[0], $l[] = random_int(1, $len - $l[0] - 3));
            $parts[] = substr($apiKey, array_sum($l));
            $apikeyPart1 = base64_encode($parts[0]);
            $apikeyPart2 = base64_encode($parts[1]);
            $apikeyPart3 = base64_encode($parts[2]);
        }

        $apiSearch = base64_encode('https://google.serper.dev/search');

        $apiSearchId = base64_encode($settings2->serper_api_key);

        if ($slug == 'ai_vision' || $slug == 'ai_pdf' || $slug == 'ai_chat_image' || $slug == 'ai_realtime_voice_chat') {

            $isPaid = false;
            $userId = Auth::user()->id;

            $activeSub = getCurrentActiveSubscription($userId);
            if ($activeSub != null) {
                $gateway = $activeSub->paid_with;
            } else {
                $activeSubY = getCurrentActiveSubscriptionYokkasa($userId);
                if ($activeSubY != null) {
                    $gateway = $activeSubY->paid_with;
                }
            }

            try {
                $isPaid = GatewaySelector::selectGateway($gateway)::getSubscriptionStatus();
            } catch (Exception $e) {
                $isPaid = false;
            }

            $category = OpenaiGeneratorChatCategory::whereSlug($slug)->firstOrFail();

            if ($isPaid == false && $category->plan == 'premium' && auth()->user()->type !== 'admin') {
                return redirect()->back()->with(['message' => __('Needs a Premium access'), 'type' => 'error']);
            }

            $list = UserOpenaiChat::where('user_id', Auth::id())->where('openai_chat_category_id', $category->id)->orderBy('is_pinned', 'desc')->orderBy('updated_at', 'desc');
            $list = $list->get();
            $chat = $list->first();
            $aiList = OpenaiGeneratorChatCategory::all();
            $lastThreeMessage = null;
            $chat_completions = null;
            if ($chat != null) {
                $lastThreeMessageQuery = $chat->messages()->whereNot('input', null)->orderBy('created_at', 'desc')->take(2);
                $lastThreeMessage = $lastThreeMessageQuery->get()->reverse();
                $category = OpenaiGeneratorChatCategory::where('id', $chat->openai_chat_category_id)->first();
                $chat_completions = str_replace(["\r", "\n"], '', $category->chat_completions) ?? null;

                if ($chat_completions != null) {
                    $chat_completions = json_decode($chat_completions, true);
                }
            }

            $models = Entity::planModels();

            $generators = OpenaiGeneratorChatCategory::query()
                ->whereNotIn('slug', [
                    'ai_vision', 'ai_webchat', 'ai_pdf',
                ])
                ->when(Auth::user()?->isUser(), function ($query) {
                    $query->where(function ($query) {
                        $query->whereNull('user_id')->orWhere('user_id', Auth::id());
                    });
                })
                ->get();

            return view('panel.user.openai_chat.chat', compact(
                'category',
                'apiSearch',
                'generators',
                'apiSearchId',
                'list',
                'chat',
                'aiList',
                'apikeyPart1',
                'apikeyPart2',
                'apikeyPart3',
                'apiUrl',
                'lastThreeMessage',
                'chat_completions',
                'models'
            ));
        }

        $view = 'panel.user.openai.generator_workbook';
        $models = Entity::planModels();
        $list = OpenAIGenerator::query()
            ->where(function ($query) {
                $query->where('user_id', Auth::id())
                    ->orWhereNull('user_id');
            })->where('active', true)->get(['id', 'title', 'slug', 'image']);

        return view($view, compact(
            'list',
            'openai',
            'apiSearch',
            'apiSearchId',
            'apikeyPart1',
            'apikeyPart2',
            'apikeyPart3',
            'apiUrl',
            'models'
        ));
    }

    public function openAIGeneratorWorkbookSave(Request $request)
    {
        $workbook = UserOpenai::where('slug', $request->workbook_slug)->where('user_id', auth()->user()->id)->firstOrFail();
        $workbook->output = $request->workbook_text;
        $workbook->title = $request->workbook_title;
        $workbook->save();

        return response()->json([], 200);
    }

    // Chat
    public function openAIChat()
    {
        $chat = Auth::user()->openaiChat;

        return view('panel.user.openai.chat', compact('chat'));
    }

    public static function sanitizeSVG($uploadedSVG)
    {

        $sanitizer = new Sanitizer;
        $content = file_get_contents($uploadedSVG);
        $cleanedData = $sanitizer->sanitize($content);
        $added = file_put_contents($uploadedSVG, $cleanedData);

        return $uploadedSVG;
    }

    // Profile user settings
    public function userSettings()
    {
        $user = Auth::user();

        return view('panel.user.settings.index', compact('user'));
    }

    public function userSettingsSave(Request $request)
    {
        $request->validate([
            'name'    => 'required|string|max:255',
            'surname' => 'required|string|max:255',
            'phone'   => 'nullable|string|max:15',
            'country' => 'nullable',
            'state'   => 'nullable|string|max:255',
            'city'    => 'nullable|string|max:255',
            'postal'  => 'nullable|string|max:255',
            'address' => 'nullable|string|max:255',
        ]);

        $user = Auth::user();
        $user->name = $request->name;
        $user->surname = $request->surname;
        $user->phone = $request->phone;
        $user->country = $request->country;
        $user->state = $request->state;
        $user->city = $request->city;
        $user->postal = $request->postal;
        $user->address = $request->address;

        if (! empty($request->old_password)) {
            $validated = $request->validateWithBag('updatePassword', [
                'old_password' => ['required', 'current_password'],
                'new_password' => ['required', Password::defaults(), 'confirmed'],
            ]);

            $user->password = Hash::make($request->new_password);
        }

        if (empty($request->old_password) && ! empty($request->new_password)) {
            $isOneOfSocialTokensNotEmpty =
                ! empty($user->google_token) || ! empty($user->github_token) || ! empty($user->facebook_token);

            if ($isOneOfSocialTokensNotEmpty) {
                $validated = $request->validateWithBag('updatePassword', [
                    'new_password' => ['required', Password::defaults(), 'confirmed'],
                ]);

                $user->password = Hash::make($request->new_password);
            }
        }

        if ($request->hasFile('avatar')) {
            $path = 'upload/images/avatar/';
            $image = $request->file('avatar');

            if ($image->guessExtension() === 'svg') {
                $image = self::sanitizeSVG($request->file('avatar'));
            }

            $image_name = Str::random(4) . '-' . Str::slug($user?->fullName()) . '-avatar.' . $image->guessExtension();

            // Image extension check
            $imageTypes = ['jpg', 'jpeg', 'png', 'svg', 'webp'];
            if (! MarketplaceHelper::isRegistered('content-manager') && ! in_array(Str::lower($image->guessExtension()), $imageTypes)) {
                $data = [
                    'errors' => ['The file extension must be jpg, jpeg, png, webp or svg.'],
                ];

                return response()->json($data, 419);
            }

            $image->move($path, $image_name);

            $user->avatar = $path . $image_name;
        }

        CreateActivity::for($user, 'Updated', 'Profile Information');
        $user->save();
    }

    public function userSettingsUpdate(Request $request): RedirectResponse
    {
        $request->validate([

            'phone'   => 'nullable|string|max:15',
            'country' => 'nullable',
            'state'   => 'nullable|string|max:255',
            'city'    => 'nullable|string|max:255',
            'postal'  => 'nullable|string|max:255',
            'address' => 'nullable|string|max:255',
        ]);

        if (! $user = Auth::user()) {
            return redirect()->back()->with(['message' => __('Unauthorized'), 'type' => 'error']);
        }

        $user->update($request->only(['phone', 'country', 'state', 'city', 'postal', 'address']));

        CreateActivity::for($user, 'Updated', 'Profile Address Information');

        return redirect()->back()->with(['message' => __('Updated Successfully'), 'type' => 'success']);
    }

    // markDashNotifySeen
    public function markDashNotifySeen()
    {
        $user = Auth::user();
        $user->dash_notify_seen = true;
        $user->save();

        return response()->json(['status' => 'success']);
    }

    // Invoice - Billing
    public function invoiceList()
    {
        $user = Auth::user();
        $list = $user?->orders;

        return view('panel.user.orders.index', compact('list'));
    }

    public function invoiceSingle($order_id)
    {
        if (auth()->user()->isAdmin()) {
            $invoice = UserOrder::where('order_id', $order_id)->firstOrFail();
        } else {
            $invoice = UserOrder::where('order_id', $order_id)->where('user_id', auth()->user()->id)->firstOrFail();
        }

        return view('panel.user.orders.invoice', compact('invoice'));
    }

    public function ordersExport($type)
    {
        $service = new OrdersExportService;

        return match ($type) {
            'pdf'   => $service->exportAsPdf(),
            'excel' => $service->exportAsExcel(),
            'csv'   => $service->exportAsCsv(),
            default => redirect()->back()->with('error', 'Invalid export type'),
        };
    }

    public function userOrdersList($user_id)
    {
        $user = User::findOrFail($user_id);
        $list = $user->orders;

        return view('panel.user.orders.index', compact('list', 'user'));
    }

    public function documentsAll(Request $request, $folderID = null)
    {
        $DOCS_PER_PAGE = 20;

        $listOnly = $request->listOnly;
        $filter = $request->filter ?? 'all';
        $sort = $request->sort ?? 'created_at';
        $sortAscDesc = $request->sortAscDesc ?? 'desc';

        // Get all documents (no filter yet)
        $documents = $this->openai($request, $folderID)
            ->where('folder_id', $folderID)
            ->where(function (Builder $query) {
                $query->whereNull('slug')
                    ->orWhere(function (Builder $query) {
                        $query->where('slug', 'not like', 'ai-image-pro-%')
                            ->where('slug', 'not like', 'ai-chat-pro-image-chat-%');
                    });
            })
            ->get();

        // Get all videos
        $videos = collect($this->getUserVideos($folderID));

        // Get AI Image Pro images
        $aiImageProImages = collect($this->getUserAiImageProImages($folderID));
        $aiChatProImageChatImages = collect($this->getUserAiChatProImageChatImages($folderID));

        // Merge everything first (use base collection concat to avoid Eloquent key-based deduplication)
        $allItems = collect($documents)
            ->concat($videos)
            ->concat($aiImageProImages)
            ->concat($aiChatProImageChatImages)
            ->values();
        // Apply filter ONCE to merged collection
        $filteredItems = $this->applyFilter($allItems, $filter);

        // Sort
        $filteredItems = $this->sortItems($filteredItems, $sort, $sortAscDesc);

        // Paginate the filtered items
        $currentPage = LengthAwarePaginator::resolveCurrentPage();
        $items = new LengthAwarePaginator(
            $filteredItems->forPage($currentPage, $DOCS_PER_PAGE),
            $filteredItems->count(),
            $DOCS_PER_PAGE,
            $currentPage,
            [
                'path'     => LengthAwarePaginator::resolveCurrentPath(),
                'pageName' => 'page',
            ]
        );

        // Preserve query parameters in pagination links
        $items->appends($request->query());

        if ($folderID !== null) {
            $currfolder = Folders::query()
                ->where(function (Builder $query) {
                    $query
                        ->where('created_by', auth()->id())
                        ->orWhere('team_id', auth()->user()->team_id);
                })
                ->findOrFail($folderID);
        } else {
            $currfolder = null;
        }

        if ($listOnly) {
            return view('panel.user.openai.documents_container', compact('items', 'currfolder', 'filter'))->render();
        }

        return view('panel.user.openai.documents', compact('items', 'currfolder', 'filter'));
    }

    protected function sortItems($items, $sort, $sortAscDesc)
    {
        $direction = $sortAscDesc === 'desc' ? 'sortByDesc' : 'sortBy';

        return match ($sort) {
            'title' => $items->$direction(fn ($item) => $item->input ?? ''),
            'created_at', 'credits' => $items->$direction($sort),
            'openai_id' => $items->$direction(fn ($item) => $item->generator->type ?? ''),
            default     => $items
        };
    }

    /**
     * Apply filtering logic to items collection
     */
    protected function applyFilter($items, $filter)
    {
        return match ($filter) {
            'favorites' => $items->filter(function ($entry) {
                if (method_exists($entry, 'isFavoriteDoc')) {
                    return $entry->isFavoriteDoc();
                }

                return (bool) ($entry->is_favorite_doc ?? false);
            }),
            'text'      => $items->filter(fn ($entry) => $entry->generator?->type === 'text'),
            'image'     => $items->filter(fn ($entry) => $entry->generator?->type === 'image'),
            'video'     => $items->filter(fn ($entry) => $entry->generator?->type === 'video'),
            'code'      => $items->filter(fn ($entry) => $entry->generator?->type === 'code'),
            'all'       => $items,
            default     => $items->filter(fn ($entry) => $entry->generator !== null),
        };
    }

    protected function getUserVideos($folderID = null): \Illuminate\Support\Collection
    {
        $user = Auth::user();
        $query = UserOpenai::query()
            ->where('user_id', $user->id)
            ->where('status', 'COMPLETED')
            ->whereNotNull('output')
            ->where('output', '!=', '')
            ->where('folder_id', $folderID)
            ->whereHas('generator', function ($q) {
                $q->where('type', 'video');
            })
            ->with(['generator' => function ($q) {
                $q->select('id', 'type', 'title');
            }]);

        $dbVideos = $query->get()->map(function ($item) {
            $item->source = 'database';

            return $item;
        });

        $userFallVideos = collect();
        if (class_exists(UserFall::class) && $folderID === null) {
            $userFallQuery = UserFall::query()
                ->where('user_id', auth()->id())
                ->where('status', 'complete')
                ->whereNotNull('video_url')
                ->where('video_url', '!=', '');

            $userFallVideos = $userFallQuery->get()->map(function ($item) {
                // Keep the original model and add properties instead of transforming to stdClass
                $item->source = 'userfall';
                $item->title = $this->generateTitleFromPrompt($item->prompt);
                $item->input = $item->prompt;
                $item->output = $item->video_url;
                $item->output_url = $item->video_url;
                $item->url = $item->video_url;
                $item->format_date = $item->created_at->format('M d, Y');
                $item->model ??= 'veo2';
                $item->is_demo = $item->is_demo ?? 0;
                $item->slug = Str::slug($item->title);
                $item->generator = (object) [
                    'id'    => null,
                    'type'  => 'video',
                    'title' => ucfirst($item->model ?? 'veo2') . ' Video',
                    'color' => null,
                    'image' => null,
                ];
                $item->is_favorite_doc = false;

                return $item;
            });
        }

        return $dbVideos->merge($userFallVideos)->sortByDesc('created_at');
    }

    protected function getUserAiImageProImages($folderID = null): \Illuminate\Support\Collection
    {
        if (
            $folderID !== null || ! MarketplaceHelper::isRegistered('ai-image-pro')
        ) {
            return collect();
        }

        $favoriteSlugs = $this->getExternalFavoriteSlugLookup('ai-image-pro-');
        $items = collect();
        $records = AiImageProModel::query()
            ->where('user_id', auth()->id())
            ->where('status', AiImageStatusEnum::COMPLETED->value)
            ->whereNotNull('generated_images')
            ->latest('created_at')
            ->get();

        foreach ($records as $record) {
            foreach (($record->generated_images ?? []) as $index => $generatedImage) {
                if ($this->isVideoUrl($generatedImage)) {
                    continue;
                }

                $item = clone $record;
                $item->source = 'ai-image-pro';
                $item->document_id = "aip-{$record->id}-{$index}";
                $item->id = "aip-{$record->id}-{$index}";
                $item->slug = "ai-image-pro-{$record->id}-{$index}";
                $item->input = $record->prompt;
                $item->output = $generatedImage;
                $item->output_url = $generatedImage;
                $item->credits = 1;
                $item->is_favorite_doc = $favoriteSlugs->has($item->slug);
                $item->generator = (object) [
                    'id'    => null,
                    'type'  => 'image',
                    'title' => 'AI Image Pro',
                    'color' => '#22c55e',
                    'image' => 'none',
                ];

                $items->push($item);
            }
        }

        return $items->sortByDesc('created_at')->values();
    }

    protected function getUserAiChatProImageChatImages($folderID = null): \Illuminate\Support\Collection
    {
        if (
            $folderID !== null || ! MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')
        ) {
            return collect();
        }

        $favoriteSlugs = $this->getExternalFavoriteSlugLookup('ai-chat-pro-image-chat-');
        $items = collect();
        $records = AiChatProImageModel::query()
            ->where('user_id', auth()->id())
            ->where('status', AiImageStatusEnum::COMPLETED->value)
            ->whereNotNull('generated_images')
            ->latest('created_at')
            ->get();

        foreach ($records as $record) {
            foreach (($record->generated_images ?? []) as $index => $generatedImage) {
                if ($this->isVideoUrl($generatedImage)) {
                    continue;
                }

                $item = clone $record;
                $item->source = 'ai-chat-pro-image-chat';
                $item->document_id = "aicpic-{$record->id}-{$index}";
                $item->id = "aicpic-{$record->id}-{$index}";
                $item->slug = "ai-chat-pro-image-chat-{$record->id}-{$index}";
                $item->input = $record->prompt;
                $item->output = $generatedImage;
                $item->output_url = $generatedImage;
                $item->credits = 1;
                $item->is_favorite_doc = $favoriteSlugs->has($item->slug);
                $item->generator = (object) [
                    'id'    => null,
                    'type'  => 'image',
                    'title' => 'AI Chat Pro Image Chat',
                    'color' => '#22c55e',
                    'image' => 'none',
                ];

                $items->push($item);
            }
        }

        return $items->sortByDesc('created_at')->values();
    }

    private function getExternalFavoriteSlugLookup(string $slugPrefix): \Illuminate\Support\Collection
    {
        return UserOpenai::query()
            ->where('user_id', auth()->id())
            ->where('slug', 'like', $slugPrefix . '%')
            ->whereHas('isFavoriteDocRelation')
            ->pluck('slug')
            ->flip();
    }

    private function parseExternalDocumentIdentifier(string $documentId): ?array
    {
        if (! preg_match('/^(aip|aicpic)-(\d+)-(\d+)$/', $documentId, $matches)) {
            return null;
        }

        $source = match ($matches[1]) {
            'aip'    => 'ai-image-pro',
            'aicpic' => 'ai-chat-pro-image-chat',
            default  => null,
        };

        if ($source === null) {
            return null;
        }

        return [
            'source'       => $source,
            'record_id'    => (int) $matches[2],
            'image_index'  => (int) $matches[3],
            'slug'         => $source . '-' . $matches[2] . '-' . $matches[3],
        ];
    }

    private function resolveExternalDocumentProxyId(array $externalDocument): ?int
    {
        $proxy = UserOpenai::query()
            ->where('user_id', auth()->id())
            ->where('slug', $externalDocument['slug'])
            ->first();

        if ($proxy) {
            return $proxy->id;
        }

        $workbook = match ($externalDocument['source']) {
            'ai-image-pro'           => $this->resolveAiImageProWorkbookBySlug($externalDocument['slug']),
            'ai-chat-pro-image-chat' => $this->resolveAiChatProImageChatWorkbookBySlug($externalDocument['slug']),
            default                  => null,
        };

        if (! $workbook) {
            return null;
        }

        $proxy = new UserOpenai;
        $proxy->user_id = auth()->id();
        $proxy->openai_id = $this->resolveExternalGeneratorId($externalDocument['source']);
        $proxy->title = $workbook->title ?? $workbook->generator?->title;
        $proxy->slug = $externalDocument['slug'];
        $proxy->input = $workbook->input ?? '';
        $proxy->response = $externalDocument['source'];
        $proxy->output = $workbook->output ?? $workbook->output_url;
        $proxy->credits = (string) ($workbook->credits ?? 1);
        $proxy->words = '0';
        $proxy->save();

        return $proxy->id;
    }

    private function resolveExternalGeneratorId(string $source): ?int
    {
        $slug = $source === 'ai-chat-pro-image-chat' ? 'ai_chat_image' : 'ai_image_generator';

        return OpenAIGenerator::query()
            ->where('slug', $slug)
            ->value('id');
    }

    private function isVideoUrl(string $url): bool
    {
        $videoExtensions = ['mp4', 'webm', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'm4v'];
        $extension = strtolower(pathinfo(parse_url($url, PHP_URL_PATH) ?? '', PATHINFO_EXTENSION));

        return in_array($extension, $videoExtensions, true);
    }

    private function generateTitleFromPrompt(?string $prompt): string
    {
        if (empty($prompt)) {
            return __('Untitled Video');
        }

        $words = explode(' ', trim($prompt));
        $titleWords = array_slice($words, 0, 6); // Take first 6 words
        $title = implode(' ', $titleWords);

        // Add ellipsis if the prompt was longer
        if (count($words) > 6) {
            $title .= '...';
        }

        return $title ?: __('Untitled Video');
    }

    protected function openai(Request $request, $folderID = null)
    {
        $team = $request->user()->getAttribute('team');

        $myCreatedTeam = $request->user()->getAttribute('myCreatedTeam');

        return UserOpenai::query()
            ->with('generator', 'isFavoriteDocRelation')
            ->where(function (Builder $query) use ($team, $myCreatedTeam) {
                $query->where('user_id', auth()->id())
                    ->when($team || $myCreatedTeam, function ($query) use ($team, $myCreatedTeam) {
                        if ($team && $team?->is_shared) {
                            $query->orWhere('team_id', $team->id);
                        }
                        if ($myCreatedTeam) {
                            $query->orWhere('team_id', $myCreatedTeam->id);
                        }
                    });
            });
    }

    public function updateFolder(Request $request, $folder)
    {
        $request->validate([
            'newFolderName' => 'required|string|max:255',
        ]);

        $folder = Folders::findOrFail($folder);
        $folder->name = $request->input('newFolderName');
        $folder->save();

        return response()->json(['message' => __('Folder name updated successfully')]);
    }

    public function updateFile(Request $request, $slug)
    {
        $request->validate([
            'newFileName' => 'required|string|max:255',
        ]);

        $file = UserOpenai::where('slug', $slug)->where('user_id', auth()->user()->id)->firstOrFail();
        $file->generator->title = $request->input('newFileName');
        $file->generator->save();

        return response()->json(['message' => __('File name updated successfully')]);
    }

    public function deleteFolder(Request $request, $folder)
    {
        $folder = Folders::findOrFail($folder);
        $all = $request->all;
        if ($all) {
            foreach ($folder->userOpenais as $userOpenai) {
                $userOpenai->delete();
            }
        }
        $folder->delete();

        return response()->json(['message' => __('Folder deleted successfully')]);
    }

    public function newFolder(Request $request)
    {
        $request->validate([
            'newFolderName' => 'required|string|max:255',
        ]);

        $newFolder = new Folders;
        $newFolder->name = $request->newFolderName;
        $newFolder->created_by = auth()->user()->id;
        $newFolder->save();

        return back()->with(['message' => __('Added successfuly'), 'type' => 'success']);
    }

    public function moveToFolder(Request $request)
    {
        $folderID = $request->selectedFolderId;
        $fileSlug = $request->fileslug;

        $workbook = UserOpenai::where('slug', $fileSlug)->where('user_id', auth()->user()->id)->firstOrFail();
        $workbook->folder_id = $folderID;
        $workbook->save();

        return back()->with(['message' => __('Moved successfuly'), 'type' => 'success']);
    }

    public function documentsSingle($slug)
    {
        $workbook = null;
        $openai = null;
        $isUserFallVideo = false;
        $isAiImagePro = false;
        $isAiChatProImageChat = false;

        $aiImageProWorkbook = $this->resolveAiImageProWorkbookBySlug($slug);
        if ($aiImageProWorkbook) {
            $workbook = $aiImageProWorkbook;
            $openai = $workbook->generator;
            $isAiImagePro = true;
        }

        $aiChatProImageChatWorkbook = $this->resolveAiChatProImageChatWorkbookBySlug($slug);
        if (! $workbook && $aiChatProImageChatWorkbook) {
            $workbook = $aiChatProImageChatWorkbook;
            $openai = $workbook->generator;
            $isAiChatProImageChat = true;
        }

        if (! $workbook) {
            $workbook = UserOpenai::where('slug', $slug)
                ->where('user_id', auth()->user()->id)
                ->first();
        }

        if ($workbook) {
            $openai = $workbook->generator;
        } elseif (class_exists(UserFall::class)) {
            $userFallVideo = UserFall::where('user_id', auth()->id())
                ->where('status', 'complete')
                ->whereNotNull('video_url')
                ->where('video_url', '!=', '')
                ->get()
                ->first(function ($item) use ($slug) {
                    // Generate title and slug the same way as in getUserVideos method
                    $title = $this->generateTitleFromPrompt($item->prompt);

                    return Str::slug($title) === $slug;
                });

            if ($userFallVideo) {
                // Transform UserFall item to match UserOpenai structure
                $workbook = $userFallVideo;
                $workbook->source = 'userfall';
                $workbook->title = $this->generateTitleFromPrompt($userFallVideo->prompt);
                $workbook->input = $userFallVideo->prompt;
                $workbook->output = $userFallVideo->video_url;
                $workbook->output_url = $userFallVideo->video_url;
                $workbook->url = $userFallVideo->video_url;
                $workbook->format_date = $userFallVideo->created_at->format('M d, Y');
                $workbook->model = $userFallVideo->model ?? 'veo2';
                $workbook->is_demo = $userFallVideo->is_demo ?? 0;
                $workbook->slug = Str::slug($workbook->title);
                $workbook->generator = (object) [
                    'id'    => null,
                    'type'  => 'video',
                    'title' => ucfirst($userFallVideo->model ?? 'veo2') . ' Video',
                    'color' => null,
                    'image' => null,
                ];
                $workbook->is_favorite_doc = false;

                $openai = $workbook->generator;
                $isUserFallVideo = true;
            }
        }

        if (! $workbook) {
            abort(404);
        }

        $integrations = Auth::user()->getAttribute('integrations');

        $wordpress = UserIntegration::query()
            ->where('user_id', Auth::user()->id)->first();

        if (isset($wordpress)) {
            $wordpressExist = (bool) $wordpress->credentials['domain']['value'];
        } else {
            $wordpressExist = false;
        }

        $checkIntegration = ($isAiImagePro || $isAiChatProImageChat) ? 0 : Integration::query()->whereHas('hasExtension')->count();

        return view('panel.user.openai.documents_workbook', compact(
            'wordpressExist',
            'checkIntegration',
            'workbook',
            'openai',
            'integrations',
            'isUserFallVideo'
        ));
    }

    public function documentsDelete($slug)
    {
        if ($this->deleteAiImageProDocument($slug)) {
            return redirect()->route('dashboard.user.openai.documents.all')
                ->with([
                    'message' => __('Document deleted successfully'),
                    'type'    => 'success',
                ]);
        }
        if ($this->deleteAiChatProImageChatDocument($slug)) {
            return redirect()->route('dashboard.user.openai.documents.all')
                ->with([
                    'message' => __('Document deleted successfully'),
                    'type'    => 'success',
                ]);
        }

        $workbook = null;
        $isUserFallVideo = false;

        // First, try to find in UserOpenai table
        $workbook = UserOpenai::where('slug', $slug)
            ->where('user_id', auth()->user()->id)
            ->first();

        if (! $workbook) {
            // If not found in UserOpenai, check UserFall table for videos
            if (class_exists(UserFall::class)) {
                $userFallVideo = UserFall::where('user_id', auth()->id())
                    ->where('status', 'complete')
                    ->whereNotNull('video_url')
                    ->where('video_url', '!=', '')
                    ->get()
                    ->first(function ($item) use ($slug) {
                        // Generate title and slug the same way as in other methods
                        $title = $this->generateTitleFromPrompt($item->prompt);

                        return Str::slug($title) === $slug;
                    });

                if ($userFallVideo) {
                    $workbook = $userFallVideo;
                    $isUserFallVideo = true;
                }
            }
        }

        // If still not found, throw 404
        if (! $workbook) {
            abort(404);
        }

        try {
            if ($isUserFallVideo) {
                // Handle UserFall video deletion
                $this->deleteUserFallVideo($workbook);
            } else {
                // Handle UserOpenai document/file deletion
                $this->deleteUserOpenaiDocument($workbook);
            }
        } catch (Throwable $th) {
            // Log error but continue with database deletion
            Log::error('Error deleting file: ' . $th->getMessage());
        }

        // Delete the database record
        $workbook->delete();

        return redirect()->route('dashboard.user.openai.documents.all')
            ->with([
                'message' => __('Document deleted successfully'),
                'type'    => 'success',
            ]);
    }

    private function deleteAiImageProDocument(string $slug): bool
    {
        if (
            ! str_starts_with($slug, 'ai-image-pro-')
            || ! MarketplaceHelper::isRegistered('ai-image-pro')
        ) {
            return false;
        }

        if (! preg_match('/^ai-image-pro-(\d+)-(\d+)$/', $slug, $matches)) {
            return false;
        }

        $recordId = (int) $matches[1];
        $imageIndex = (int) $matches[2];

        $record = AiImageProModel::query()
            ->where('user_id', auth()->id())
            ->find($recordId);

        if (! $record) {
            return false;
        }

        $images = $record->generated_images ?? [];

        if (! array_key_exists($imageIndex, $images)) {
            return false;
        }

        unset($images[$imageIndex]);

        if (empty($images)) {
            $record->delete();
        } else {
            $record->generated_images = $images;
            $record->save();
        }

        UserOpenai::query()
            ->where('user_id', auth()->id())
            ->where('slug', $slug)
            ->delete();

        return true;
    }

    private function resolveAiImageProWorkbookBySlug(string $slug): ?AiImageProModel
    {
        if (
            ! str_starts_with($slug, 'ai-image-pro-')
            || ! MarketplaceHelper::isRegistered('ai-image-pro')
        ) {
            return null;
        }

        if (! preg_match('/^ai-image-pro-(\d+)-(\d+)$/', $slug, $matches)) {
            return null;
        }

        $recordId = (int) $matches[1];
        $imageIndex = (int) $matches[2];

        $record = AiImageProModel::query()
            ->where('user_id', auth()->id())
            ->find($recordId);

        if (! $record) {
            return null;
        }

        $images = $record->generated_images ?? [];
        if (! array_key_exists($imageIndex, $images)) {
            return null;
        }

        $workbook = clone $record;
        $workbook->source = 'ai-image-pro';
        $workbook->id = "aip-{$record->id}-{$imageIndex}";
        $workbook->slug = $slug;
        $workbook->title = 'AI Image Pro';
        $workbook->input = $record->prompt;
        $workbook->output = $images[$imageIndex];
        $workbook->output_url = $images[$imageIndex];
        $workbook->credits = 1;
        $workbook->generator = (object) [
            'id'    => null,
            'type'  => 'image',
            'title' => 'AI Image Pro',
            'color' => '#22c55e',
            'image' => 'none',
        ];

        return $workbook;
    }

    private function deleteAiChatProImageChatDocument(string $slug): bool
    {
        if (
            ! str_starts_with($slug, 'ai-chat-pro-image-chat-')
            || ! MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')
        ) {
            return false;
        }

        if (! preg_match('/^ai-chat-pro-image-chat-(\d+)-(\d+)$/', $slug, $matches)) {
            return false;
        }

        $recordId = (int) $matches[1];
        $imageIndex = (int) $matches[2];

        $record = AiChatProImageModel::query()
            ->where('user_id', auth()->id())
            ->find($recordId);

        if (! $record) {
            return false;
        }

        $images = $record->generated_images ?? [];
        if (! array_key_exists($imageIndex, $images)) {
            return false;
        }

        unset($images[$imageIndex]);

        if (empty($images)) {
            $record->delete();
        } else {
            $record->generated_images = $images;
            $record->save();
        }

        UserOpenai::query()
            ->where('user_id', auth()->id())
            ->where('slug', $slug)
            ->delete();

        return true;
    }

    private function resolveAiChatProImageChatWorkbookBySlug(string $slug): ?AiChatProImageModel
    {
        if (
            ! str_starts_with($slug, 'ai-chat-pro-image-chat-')
            || ! MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')
        ) {
            return null;
        }

        if (! preg_match('/^ai-chat-pro-image-chat-(\d+)-(\d+)$/', $slug, $matches)) {
            return null;
        }

        $recordId = (int) $matches[1];
        $imageIndex = (int) $matches[2];

        $record = AiChatProImageModel::query()
            ->where('user_id', auth()->id())
            ->find($recordId);

        if (! $record) {
            return null;
        }

        $images = $record->generated_images ?? [];
        if (! array_key_exists($imageIndex, $images)) {
            return null;
        }

        $workbook = clone $record;
        $workbook->source = 'ai-chat-pro-image-chat';
        $workbook->id = "aicpic-{$record->id}-{$imageIndex}";
        $workbook->slug = $slug;
        $workbook->title = 'AI Chat Pro Image Chat';
        $workbook->input = $record->prompt;
        $workbook->output = $images[$imageIndex];
        $workbook->output_url = $images[$imageIndex];
        $workbook->credits = 1;
        $workbook->generator = (object) [
            'id'    => null,
            'type'  => 'image',
            'title' => 'AI Chat Pro Image Chat',
            'color' => '#22c55e',
            'image' => 'none',
        ];

        return $workbook;
    }

    public function documentsBulkDelete(Request $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json(['success' => false, 'message' => __('This action is disabled in the demo mode.')], 403);
        }

        $userId = auth()->id();

        try {
            // Case: Delete all
            if ($request->boolean('all')) {
                // Delete UserOpenai docs
                $docs = UserOpenai::where('user_id', $userId)->get();
                foreach ($docs as $doc) {
                    try {
                        $this->deleteUserOpenaiDocument($doc);
                    } catch (Throwable $e) {
                        Log::error("Error deleting doc {$doc->id}: " . $e->getMessage());
                    }
                    $doc->delete();
                }

                // Delete UserFall videos if applicable
                if (class_exists(UserFall::class)) {
                    $videos = UserFall::where('user_id', $userId)
                        ->where('status', 'complete')
                        ->whereNotNull('video_url')
                        ->where('video_url', '!=', '')
                        ->get();

                    foreach ($videos as $video) {
                        try {
                            $this->deleteUserFallVideo($video);
                        } catch (Throwable $e) {
                            Log::error("Error deleting video {$video->id}: " . $e->getMessage());
                        }
                        $video->delete();
                    }
                }

                if (MarketplaceHelper::isRegistered('ai-image-pro')) {
                    AiImageProModel::query()
                        ->where('user_id', $userId)
                        ->delete();
                }
                if (MarketplaceHelper::isRegistered('ai-chat-pro-image-chat')) {
                    AiChatProImageModel::query()
                        ->where('user_id', $userId)
                        ->delete();
                }

                return response()->json([
                    'success' => true,
                    'message' => __('All documents deleted successfully.'),
                ]);
            }

            // Case: Delete selected
            $ids = $request->input('ids', []);
            if (! is_array($ids) || empty($ids)) {
                return response()->json(['success' => false, 'message' => __('No documents selected.')], 400);
            }

            foreach ($ids as $id) {
                if (! is_string($id)) {
                    continue;
                }

                $externalDocument = $this->parseExternalDocumentIdentifier($id);
                if (! $externalDocument) {
                    continue;
                }

                if ($externalDocument['source'] === 'ai-image-pro') {
                    $this->deleteAiImageProDocument($externalDocument['slug']);
                }

                if ($externalDocument['source'] === 'ai-chat-pro-image-chat') {
                    $this->deleteAiChatProImageChatDocument($externalDocument['slug']);
                }
            }

            $numericIds = collect($ids)
                ->filter(fn ($id) => is_numeric($id))
                ->map(fn ($id) => (int) $id)
                ->values()
                ->all();

            // Delete UserOpenai documents
            $docs = UserOpenai::where('user_id', $userId)
                ->whereIn('id', $numericIds)
                ->get();

            foreach ($docs as $doc) {
                try {
                    $this->deleteUserOpenaiDocument($doc);
                } catch (Throwable $e) {
                    Log::error("Error deleting doc {$doc->id}: " . $e->getMessage());
                }
                $doc->delete();
            }

            // Delete UserFall videos (if selected IDs match)
            if (class_exists(UserFall::class)) {
                $videos = UserFall::where('user_id', $userId)
                    ->whereIn('id', $numericIds)
                    ->get();

                foreach ($videos as $video) {
                    try {
                        $this->deleteUserFallVideo($video);
                    } catch (Throwable $e) {
                        Log::error("Error deleting video {$video->id}: " . $e->getMessage());
                    }
                    $video->delete();
                }
            }

            return response()->json([
                'success' => true,
                'message' => __('Selected documents deleted successfully.'),
            ]);

        } catch (Throwable $th) {
            Log::error('Bulk delete failed: ' . $th->getMessage());

            return response()->json(['success' => false, 'message' => __('Something went wrong.')], 500);
        }
    }

    /**
     * Delete UserOpenai document files
     */
    private function deleteUserOpenaiDocument($workbook)
    {
        if ($workbook->storage == UserOpenai::STORAGE_LOCAL) {
            $file = str_replace('/uploads/', '', $workbook->output);
            Storage::disk('public')->delete($file);
        } elseif ($workbook->storage == UserOpenai::STORAGE_AWS) {
            $file = str_replace('/', '', parse_url($workbook->output)['path']);
            Storage::disk('s3')->delete($file);
        } else {
            // Manual deleting depends on response
            if (str_contains($workbook->output, 'https://')) {
                // AWS Storage
                $file = str_replace('/', '', parse_url($workbook->output)['path']);
                Storage::disk('s3')->delete($file);
            } else {
                $file = str_replace('/uploads/', '', $workbook->output);
                Storage::disk('public')->delete($file);
            }
        }

        // Delete thumbnail if exists
        $basefilename = basename($workbook->output);
        Storage::disk('thumbs')->delete($basefilename);
    }

    /**
     * Delete UserFall video files
     */
    private function deleteUserFallVideo($userFallVideo)
    {
        // Check if video_url contains a file that needs to be deleted
        if ($userFallVideo->video_url && ! empty($userFallVideo->video_url)) {
            // Check if it's a local file or external URL
            if (str_contains($userFallVideo->video_url, 'https://')) {
                // If it's an external URL (like AWS S3), try to delete it
                $parsedUrl = parse_url($userFallVideo->video_url);
                if (isset($parsedUrl['path'])) {
                    $file = ltrim($parsedUrl['path'], '/');
                    Storage::disk('s3')->delete($file);
                }
            } else {
                // If it's a local file
                $file = str_replace('/uploads/', '', $userFallVideo->video_url);
                Storage::disk('public')->delete($file);
            }

            // Delete thumbnail if exists
            $basefilename = basename($userFallVideo->video_url);
            Storage::disk('thumbs')->delete($basefilename);
        }
    }

    public function documentsImageDelete($slug)
    {
        $workbook = UserOpenai::where('slug', $slug)->where('user_id', auth()->user()->id)->firstOrFail();

        try {
            if ($workbook->storage == UserOpenai::STORAGE_LOCAL) {
                $file = str_replace('/uploads/', '', $workbook->output);
                Storage::disk('public')->delete($file);
            } elseif ($workbook->storage == UserOpenai::STORAGE_AWS) {
                $file = str_replace('/', '', parse_url($workbook->output)['path']);
                Storage::disk('s3')->delete($file);
            } elseif ($workbook->storage == UserOpenai::STORAGE_R2) {
                if (str_contains($workbook->output, 'https://')) {
                    // AWS Storage
                    $file = str_replace('/', '', parse_url($workbook->output)['path']);

                    Storage::disk('r2')->delete($file);
                }
            } else {

                // Manual deleting depends on response
                if (str_contains($workbook->output, 'https://')) {
                    // AWS Storage
                    $file = str_replace('/', '', parse_url($workbook->output)['path']);
                    Storage::disk('s3')->delete($file);
                } else {
                    $file = str_replace('/uploads/', '', $workbook->output);
                    Storage::disk('public')->delete($file);
                }

            }

            $basefilename = basename($workbook->output);
            $workbook->delete();
            Storage::disk('thumbs')->delete($basefilename);
        } catch (Exception $exception) {
        }

        $workbook->delete();

        return back()->with(['message' => __('Deleted successfuly'), 'type' => 'success']);
    }

    // Affiliates
    public function affiliatesList()
    {
        $user = Auth::user();
        abort_if(Helper::setting('feature_affilates') == 0, 404);
        if (setting('affiliate_plan_restriction', '0')) {
            abort_if(! check_plan_affiliate_status($user), 404);
        }

        $onetimeCommission = setting('onetime_commission', 0);
        $list = $user?->affiliates;
        $list2 = $user?->withdrawals;
        $totalEarnings = 0;
        foreach ($list as $affUser) {
            if ($onetimeCommission) {
                // if one time commission is open then get only the first order
                $totalEarnings += $affUser->orders->sortBy('id')->first()?->affiliate_earnings;
            } else {
                $totalEarnings += $affUser->orders->sum('affiliate_earnings');
            }
        }
        $totalWithdrawal = 0;
        foreach ($list2 as $affWithdrawal) {
            $totalWithdrawal += $affWithdrawal->amount;
        }

        return view('panel.user.affiliate.index', compact('list', 'list2', 'totalEarnings', 'totalWithdrawal'));
    }

    public function affiliatesUsers(Request $request)
    {
        $setting = Setting::getCache();

        $defaultCurrency = Currency::find($setting->default_currency)->symbol;

        $query = User::where('affiliate_id', auth()->user()->id);

        if ($request->has('search')) {
            $searchTerm = $request->input('search');
            $query->where('name', 'like', '%' . $searchTerm . '%');
        }

        if ($request->has('startDate') && $request->input('startDate')) {
            $query->whereDate('created_at', '>=', $request->input('startDate'));
        }

        if ($request->has('endDate') && $request->input('endDate')) {
            $query->whereDate('created_at', '<=', $request->input('endDate'));
        }

        $list = $query->paginate(10);

        foreach ($list as $user) {
            $affiliate = UserAffiliate::where('user_id', $user->id)->first();
            if ($affiliate) {
                $user->affiliate_data = $affiliate;
            }
        }

        return view('panel.user.affiliate.users', compact(['list', 'defaultCurrency']));
    }

    public function affiliatesListSendInvitation(Request $request)
    {
        $user = Auth::user();

        $sendTo = $request->to_mail;

        dispatch(new SendInviteEmail($user, $sendTo));

        return response()->json([], 200);
    }

    public function affiliatesListSendRequest(Request $request)
    {
        $user = Auth::user();
        $list = $user->affiliates;
        $list2 = $user->withdrawals;

        $totalEarnings = 0;
        foreach ($list as $affOrders) {
            $totalEarnings += $affOrders->orders->sum('affiliate_earnings');
        }
        $totalWithdrawal = 0;
        foreach ($list2 as $affWithdrawal) {
            $totalWithdrawal += $affWithdrawal->amount;
        }
        if ($totalEarnings - $totalWithdrawal >= $request->amount) {
            $user->affiliate_bank_account = $request->affiliate_bank_account;
            $user->save();
            $withdrawalReq = new UserAffiliate;
            $withdrawalReq->user_id = Auth::id();
            $withdrawalReq->amount = $request->amount;
            $withdrawalReq->save();

            CreateActivity::for($user, 'Sent', 'Affiliate Withdraw Request', route('dashboard.admin.affiliates.index'));
        } else {
            return response()->json(['error' => __('ERROR')], 411);
        }
    }

    public function apiKeysList()
    {
        abort_if(
            Helper::appIsNotDemo() &&
            (
                (int) Helper::setting('user_api_option') === 0
                &&
                ! auth()->user()?->relationPlan?->getAttribute('user_api')
            ), 404);

        $user = Auth::user();
        $list = $user?->api_keys;
        $anthropic_api_keys = $user?->anthropic_api_keys;
        $gemini_api_keys = $user?->gemini_api_keys;

        return view('panel.user.apiKeys.index', compact('list', 'anthropic_api_keys', 'gemini_api_keys'));
    }

    public function apiKeysSave(Request $request)
    {
        if (Helper::appIsDemo()) {
            if ($request->ajax()) {
                return response()->json([
                    'message' => __('This feature is disabled in Demo version.'),
                    'type'    => 'success',
                ], 200);
            }

            return back()->with(['message' => __('This feature is disabled in Demo version.'), 'type' => 'error']);
        }

        $user = Auth::user();
        if ($user) {
            $user->api_keys = $request->api_keys;
            $user->anthropic_api_keys = $request->anthropic_api_keys;
            $user->gemini_api_keys = $request->gemini_api_keys;
            $user->save();
        }

        if ($request->ajax()) {
            return response()->json([
                'message' => __('Settings saved successfully.'),
                'type'    => 'success',
            ], 200);
        }

        return redirect()->back();
    }

    public function overview()
    {
        $overviewData = [
            [
                'title' => 'Image Documents',
                'slug'  => 'image_documents',
                'count' => 0,
            ],
            [
                'title' => 'Code Documents',
                'slug'  => 'code_documents',
                'count' => 0,
            ],
            [
                'title' => 'Other Documents',
                'slug'  => 'other_documents',
                'count' => 0,
            ],
        ];
        $total = 0;

        $userId = Auth::id();
        $userOpenai = UserOpenai::where('user_id', $userId)->with('generatorWithType')->get();
        $imageCount = 0;
        $codeCount = 0;
        $otherCount = 0;
        foreach ($userOpenai as $key => $value) {
            if ($value->generator_type == 'image') {
                $imageCount++;
            } elseif ($value->generator_type == 'code') {
                $codeCount++;
            } else {
                $otherCount++;
            }
            $total++;
        }
        $overviewData[0]['count'] = $imageCount;
        $overviewData[1]['count'] = $codeCount;
        $overviewData[2]['count'] = $otherCount;

        $percantageOther = round(($otherCount / $total) * 100);

        return response()->json(['data' => $overviewData, 'total' => $total, 'percantageOther' => $percantageOther]);
    }

    public function deleteAccount()
    {
        return view('panel.user.settings.deleteAccount');
    }

    public function deleteAccountRequest(Request $request)
    {
        abort_if(Helper::appIsDemo(), 404);
        $request->validate([
            'password' => 'required',
        ]);
        $user = Auth::user();
        if (Hash::check($request->password, $user->password)) {
            $oldRequest = AccountDeletionReqs::where('user_id', $user->id)->first();
            if ($oldRequest) {
                return response()->json(['message' => __('You have already requested to delete your account')], 409);
            }
            $deletionRequest = new AccountDeletionReqs;
            $deletionRequest->user_id = $user->id;
            $deletionRequest->save();

            return response()->json(['message' => __('Your account deletion request has been successfully submitted')], 200);
        }

        return response()->json(['message' => __('Password is incorrect')], 401);
    }

    public function exportInvoices(Request $request)
    {
        $type = 'pdf';
        $cleanDates = static function ($date) {
            return preg_replace('/(\+|\-)?(\d{2})(\d{2})$/', '+$2:$3', str_replace('GMT ', '', preg_replace('/\s*\(.*\)$/', '', $date)));
        };
        $startDate = Carbon::createFromFormat('D M d Y H:i:s O', $cleanDates($request->start_date));
        $endDate = Carbon::createFromFormat('D M d Y H:i:s O', $cleanDates($request->end_date));
        $invoices = UserOrder::whereBetween('created_at', [$startDate, $endDate])->get();
        $service = new OrdersExportService;

        return match ($type) {
            'pdf'   => $service->exportAsPdf($invoices),
            'excel' => $service->exportAsExcel(),
            'csv'   => $service->exportAsCsv(),
            default => redirect()->back()->with('error', 'Invalid export type'),
        };
    }

    /**
     * check if version or extension update is available
     */
    public function updateAvailable(ExtensionRepositoryInterface $extensionRepository): JsonResponse
    {
        $cacheKey = 'system.update.status';

        $data = Cache::remember($cacheKey, now()->addMinutes(10), function () use ($extensionRepository) {
            $versionUpdateAvailable = false;
            $extensionUpdateAvailable = false;
            $updateAvailableExtensions = [];

            if (Updater::versionCheck()) {
                $extensionItems = $extensionRepository->extensions();
                $updateAvailableExtensions = array_filter(
                    $extensionItems,
                    static fn ($item) => ($item['installed'] ?? false) && ($item['upgradable'] ?? false)
                );
                $extensionUpdateAvailable = (bool) count($updateAvailableExtensions);
            } else {
                $versionUpdateAvailable = true;
            }

            return compact('versionUpdateAvailable', 'extensionUpdateAvailable', 'updateAvailableExtensions');
        });

        return response()->json($data);
    }
}

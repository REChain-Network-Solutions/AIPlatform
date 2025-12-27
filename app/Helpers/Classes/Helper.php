<?php

namespace App\Helpers\Classes;

use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use App\Enums\Roles;
use App\Helpers\Classes\RateLimiter\RateLimiter;
use App\Models\Currency;
use App\Models\Finance\Subscription;
use App\Models\Setting;
use App\Models\SettingTwo;
use DateInterval;
use DatePeriod;
use DateTimeImmutable;
use Exception;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Route;
use Spatie\Permission\Models\Role;

class Helper
{
    use Traits\HasApiKeys;

    public static function octaneReload(): void
    {
        try {
            shell_exec('cd ' . base_path() . ' && php artisan octane:reload 2>&1');
        } catch (Exception $exception) {
        }
    }

    public static function adminPermissions(string $key)
    {
        $permissions = Cache::remember('admin_permissions', 3600 * 24, function () {
            $role = Role::findByName(Roles::ADMIN->value);

            return collect($role->getAllPermissions())->pluck('name')->merge(['admin_dashboard']);
        });

        return $permissions->contains($key);
    }

    public static function getCurrentActiveSubscription($userId = null)
    {
        if (self::appIsDemo()) {
            return null;
        }

        $userId = $userId ?? Auth::id();

        return once(static function () use ($userId) {
            return Subscription::query()
                ->with('plan')
                ->where('user_id', $userId)
                ->whereIn('stripe_status', [
                    'active',
                    'trialing',
                    'bank_approved',
                    'banktransfer_approved',
                    'bank_renewed',
                    'free_approved',
                    'stripe_approved',
                    'paypal_approved',
                    'iyzico_approved',
                    'paystack_approved',
                ])
                ->first();
        });
    }

    public static function showIntercomForBuyer(array $extensions = []): bool
    {
        if (self::showIntercomForVipMembership()) {
            return false;
        }

        $extensionsCollection = collect(
            $extensions ?: app(ExtensionRepositoryInterface::class)->extensions()
        )
            ->where('is_theme', request()?->routeIs('dashboard.admin.themes.*'))
            ->where('licensed', true)
            ->where('price', '>', 0);

        return $extensionsCollection->isNotEmpty();
    }

    public static function isUserVIP()
    {
        return Cache::remember('vip_membership', 300, function () {
            $marketSubscription = app(ExtensionRepositoryInterface::class)->subscription()->json();

            return data_get($marketSubscription, 'data');
        });
    }

    public static function showIntercomForVipMembership(): bool
    {
        if (self::appIsDemo()) {
            return false;
        }

        $user = Auth::user();

        if (! $user?->isAdmin()) {
            return false;
        }

        if (! $user->isSuperAdmin() && ! $user->checkPermission('VIP_CHAT_WIDGET')) {
            return false;
        }

        return (bool) self::isUserVIP();
    }

    public static function marketplacePaymentMessage(string $status): string
    {
        return match ($status) {
            'paid'    => __('Your payment has been received successfully.'),
            'pending' => __('Your payment is pending. Please check back later for confirmation. Once processed, you\'ll be able to download the extension.'),
            default   => __('Your payment was unsuccessful. Please try again'),
        };
    }

    public static function hasRoute($route = null): bool
    {
        return $route && Route::has($route);
    }

    public static function arrayMerge($merge = true, array $array = [], array ...$mergeArray): array
    {
        if ($merge) {
            return array_merge($array, ...$mergeArray);
        }

        return $array;
    }

    public static function decodePaymentToken(string $token): array
    {
        $base64 = base64_decode($token);

        $data = explode(':', $base64);

        [$domain, $slug, $license] = $data;

        return compact('domain', 'slug', 'license');
    }

    public static function setEnv(array $data = []): void
    {
        $envPath = base_path('.env');

        $content = file_get_contents($envPath);

        foreach ($data as $key => $value) {
            $keyPattern = preg_quote($key, '/');
            $pattern = "/^{$keyPattern}=[^\r\n]*/m";
            if (preg_match($pattern, $content)) {
                $content = preg_replace($pattern, "{$key}=" . '"' . "{$value}" . '"', $content);
            } else {
                $content .= "\n{$key}=" . '"' . "{$value}" . '"';
            }
        }

        file_put_contents($envPath, $content);
    }

    public static function dbConnectionStatus(): bool
    {
        try {
            DB::connection()->getPdo();

            return true;
        } catch (Exception $e) {
            return false;
        }
    }

    public static function sortingOpenAiSelected($data, $selected = null)
    {
        $data = $data->map(function ($item) use ($selected) {
            $item->selected = 0;
            if ($selected && in_array($item->slug, $selected, true)) {
                $item->selected = 1;
            }

            return $item;
        });

        return $data->sortByDesc('selected');
    }

    public static function strip_all_tags($text, $remove_breaks = false)
    {
        if (is_null($text)) {
            return '';
        }

        if (! is_scalar($text)) {
            /*
             * To maintain consistency with pre-PHP 8 error levels,
             * trigger_error() is used to trigger an E_USER_WARNING,
             * rather than _doing_it_wrong(), which triggers an E_USER_NOTICE.
             */
            trigger_error(
                sprintf(
                    /* translators: 1: The function name, 2: The argument number, 3: The argument name, 4: The expected type, 5: The provided type. */
                    __('Warning: %1$s expects parameter %2$s (%3$s) to be a %4$s, %5$s given.'),
                    __FUNCTION__,
                    '#1',
                    '$text',
                    'string',
                    gettype($text)
                ),
                E_USER_WARNING
            );

            return '';
        }

        $text = preg_replace('@<(script|style)[^>]*?>.*?</\\1>@si', '', $text);
        $text = strip_tags($text);

        if ($remove_breaks) {
            $text = preg_replace('/[\r\n\t ]+/', ' ', $text);
        }

        return trim($text);
    }

    public static function multi_explode($delimiters, $string): array
    {

        $ready = str_replace($delimiters, $delimiters[0], $string);

        $ready = str_replace([',,'], ',', $ready);

        $ready = str_replace('-', '', $ready);

        return explode($delimiters[0], $ready);
    }

    public static function settingTwo(string $key, $default = null)
    {
        $setting = SettingTwo::getCache();

        return $setting?->getAttribute($key) ?? $default;
    }

    public static function setting(string $key, $default = null, $setting = null)
    {
        $setting = $setting ?: Setting::getCache();

        return $setting?->getAttribute($key) ?? $default;
    }

    public static function appIsDemoForChatbot(): bool
    {
        return self::appIsDemo() && in_array(request()->getHost(), ['magicai.liquid-themes.com', 'demo.magicproject.ai']);
    }

    public static function appIsDemo(): bool
    {
        return config('app.status') === 'Demo';
    }

    public static function appIsNotDemo(): bool
    {
        return config('app.status') !== 'Demo';
    }

    public static function generateFakeDataNewCustomer(): array
    {
        $days = collect();

        for ($i = 9; $i >= 0; $i--) {
            $date = now()->subDays($i)->format('Y-m-d');
            $days->put($date, [
                'date' => $date,
                'paid' => rand(0, 5),
                'free' => rand(0, 3),
            ]);
        }

        return $days->toArray();
    }

    public static function generateFakeDataLastMonth(): array
    {
        $now = new DateTimeImmutable; // Şu anki tarih
        $startDate = $now->modify('-1 month'); // 1 ay öncesi
        $interval = new DateInterval('P1D'); // Günlük adım
        $dateRange = new DatePeriod($startDate, $interval, $now->modify('+1 day')); // 1 ay boyunca

        $data = [];

        foreach ($dateRange as $date) {
            $data[] = [
                'sums' => rand(0, 100), // İstersen bu aralığı değiştirebilirsin
                'days' => $date->format('Y-m-d'),
            ];
        }

        return $data;
    }

    public static function demoDataForAdminDashboardPopularPlans(): array
    {
        return [
            [
                'label' => 'Lifetime',
                'value' => 67,
                'color' => '#4CAF50', // Yeşil
            ],
            [
                'label' => 'Yearly',
                'value' => 83,
                'color' => '#2196F3', // Mavi
            ],
            [
                'label' => 'Monthly',
                'value' => 280,
                'color' => '#FFC107', // Amber / Sarımsı
            ],
            [
                'label' => 'Prepaid',
                'value' => 120,
                'color' => '#F44336', // Kırmızı
            ],
        ];
    }

    public static function demoDataForAdminDashboardTopCountries(): false|string
    {
        $sites = [
            [
                'country'  => 'United Kingdom',
                'total'    => 2132,
            ],
            [
                'country'  => 'Germany',
                'total'    => 754,
            ],
            [
                'country'  => 'Brazil',
                'total'    => 244,
            ],
            [
                'country'  => 'US',
                'total'    => 232,
            ],
        ];

        return json_encode($sites);
    }

    public static function demoDataForAdminDashboardUserTraffic(): false|string
    {
        //			'google.com', 'codecanyon.com', 'bing.com', 'yandex.com', 'openai.com'
        $sites = [
            [
                'users'  => 1232,
                'domain' => 'google.com',
            ],
            [
                'users'  => 3423,
                'domain' => 'codecanyon.com',
            ],
            [
                'users'  => 423,
                'domain' => 'bing.com',
            ],
            [
                'users'  => 2323,
                'domain' => 'yandex.com',
            ],
            [
                'users'  => 350,
                'domain' => 'openai.com',
            ],
        ];

        return json_encode($sites);
    }

    public static function checkImageDailyLimit($lockKey = 'default_lock_key'): JsonResponse
    {
        $settings_two = SettingTwo::getCache();

        if ($settings_two->daily_limit_enabled) {
            if (self::appIsDemo()) {
                $msg = __('You have reached the maximum number of image generation allowed on the demo.');
            } else {
                $msg = __('You have reached the maximum number of image generation allowed.');
            }

            if (! Cache::lock($lockKey, 10)->get()) { // Attempt to acquire lock
                return response()->json(['message' => 'Image generation in progress. Please try again later.'], 409);
            }

            $clientIp = self::getRequestIp();
            $rateLimiter = new RateLimiter('ai_image_generator_rate_limit', 3);

            if (! $rateLimiter->attempt($clientIp)) {
                return response()->json([
                    'status'  => 'error',
                    'message' => $msg,
                ], 429);
            }
        }

        return response()->json([], 200);
    }

    public static function checkDemoSecondDailyLimit($incomingSeconds = 0, $lockKey = 'default_second_lock_key'): JsonResponse
    {
        if (! self::appIsDemo()) {
            return response()->json([], 200);
        }

        $msg = __('You have reached the maximum allowed AI generation time for today (30 seconds). Please try again tomorrow.');

        // Prevent concurrent requests from interfering
        if (! Cache::lock($lockKey, 10)->get()) {
            return response()->json(['message' => 'Request in progress. Please try again later.'], 409);
        }

        $clientIp = self::getRequestIp();
        $cacheKey = "demo_ai_usage_seconds_{$clientIp}";
        $maxSecondsPerDay = 30;

        // Get how many seconds this IP has used today
        $usedSeconds = Cache::get($cacheKey, 0);

        // If adding the new request would exceed limit
        if (($usedSeconds + $incomingSeconds) > $maxSecondsPerDay) {
            Cache::lock($lockKey)->release();

            return response()->json([
                'status'  => 'error',
                'message' => $msg,
            ], 429);
        }

        // Release the lock (no update yet — we only update after successful generation)
        Cache::lock($lockKey)->release();

        return response()->json([
            'status'            => 'ok',
            'remaining_seconds' => $maxSecondsPerDay - $usedSeconds,
        ], 200);
    }

    public static function checkRemainingImages($usr = null)
    {
        $user = $usr ?? auth()->user();
        if ($user->getAttribute('team')) {
            $teamManager = $user->teamManager;
            if ($teamManager) {
                if ($teamManager->remaining_images <= 0 and $teamManager->remaining_images != -1) {
                    $data = [
                        'errors' => [__('You have no credits left. Please consider upgrading your plan.')],
                    ];

                    return response()->json($data, 429);
                }
            }
            $member = $user->teamMember;
            if ($member) {
                if (! $member->allow_unlimited_credits) {
                    if ($member->remaining_images <= 0 and $member->remaining_images != -1) {
                        $data = [
                            'errors' => [__('You have no credits left. Please consider upgrading your plan.')],
                        ];

                        return response()->json($data, 429);
                    }
                }
            }
        } else {
            if ($user->remaining_images <= 0 and $user->remaining_images != -1) {
                $data = [
                    'errors' => [__('You have no credits left. Please consider upgrading your plan.')],
                ];

                return response()->json($data, 429);
            }
        }

        return response()->json([], 200);
    }

    public static function sorting(array $data, $column, $direction): array|Collection
    {
        if ($column) {
            $data = collect($data)->sortBy($column, SORT_REGULAR, $direction == 'asc');
        }

        return $data;
    }

    public static function parseUrl(string ...$urls): string
    {
        return collect($urls)->map(fn ($url) => trim($url, '/'))->implode('/');
    }

    public static function findCurrencyFromId(null|int|string $id)
    {
        return Currency::cacheFromSetting((string) $id);
    }

    public static function defaultCurrency($column = 'code')
    {
        $currency = Currency::query()->where('id', Helper::setting('default_currency'))->first();

        if (is_null($column)) {
            return $currency;
        }

        return $currency->$column;
    }

    public static function generateNumberForDemo($default = '---')
    {
        return self::appIsDemo() ? random_int(1, 100) : $default;
    }

    public static function getRequestCountryCode(): ?string
    {
        $request = request();

        if ($request->hasHeader('cf-ipcountry')) {
            return $request->headers->get('cf-ipcountry');
        }

        return null;
    }

    public static function getRequestIp(): string
    {
        $request = request();

        return $request->header('CF-Connecting-IP')
            ?? $request->header('X-Forwarded-For')
            ?? $request->header('X-Real-IP')
            ?? $request->ip();
    }

    public static function clearEmptyConversations(): void
    {
        $chats = Auth::user()?->openaiChat()->where('is_empty', true)->get();
        foreach ($chats ?? [] as $chat) {
            $chat?->messages()?->delete();
            $chat?->delete();
        }
    }
}

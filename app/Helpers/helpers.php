<?php

use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Finance\PaymentProcessController;
use App\Models\Coupon;
use App\Models\Currency;
use App\Models\Finance\YokassaSubscription;
use App\Models\Gateways;
use App\Models\PaymentProof;
use App\Models\Plan;
use App\Models\PrivacyTerms;
use App\Models\UserOrder;
use App\Services\Chatbot\ParserExcelService;
use Datlechin\GoogleTranslate\Facades\GoogleTranslate;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

function formatSmallNumber($value, $decimals = 10)
{
    return rtrim(rtrim(sprintf("%.{$decimals}f", $value), '0'), '.');
}

function getImageUrlByOrderId($orderId)
{
    $paymentProof = PaymentProof::where('order_id', $orderId)->first();

    if ($paymentProof) {
        return asset('proofs/' . $paymentProof->proof_image);
    }

    return null;
}

function bankActive()
{
    return Gateways::getCache(
        static fn () => Gateways::query()->where('code', 'banktransfer')->where('is_active', 1)->exists(),
        '_banktransfer_active'
    );
}

function countBankTansactions()
{
    $bankOrdersCount = UserOrder::where('payment_type', 'banktransfer')->where('status', 'Waiting')->count();

    return $bankOrdersCount;
}

function getCurrentActiveSubscription($userId = null)
{
    return Helper::getCurrentActiveSubscription($userId);
}

function formatPrice($price)
{
    return $price == intval($price) ? number_format($price, 0) : number_format($price, 2);
}

function getCurrentActiveSubscriptionYokkasa($userid = null)
{
    if (Helper::appIsDemo()) {
        return null;
    }

    $userID = $userid ?? auth()->user()?->id;

    $cacheKey = 'yokassa_subscription_' . ($userID ? 'user_' . $userID : 'guest');

    return Cache::remember($cacheKey, 3, static function () use ($userID) {
        $yokassaActive = Gateways::getCache(
            static fn () => Gateways::query()
                ->where(['code' => 'yokassa', 'is_active' => 1])
                ->first(),
            '_active_yokassa'
        );

        if (! $yokassaActive) {
            return null;
        }

        return YokassaSubscription::query()
            ->where('user_id', $userID)
            ->whereIn('subscription_status', ['active', 'yokassa_approved'])
            ->first();
    });
}

function getTokenPlans()
{
    return Plan::getCache(
        static fn () => Plan::query()->where('type', 'prepaid')->where('active', 1)->get(),
        '_prepaid_plans'
    );
}
function getSubsPlans()
{
    return Plan::getCache(
        static fn () => Plan::query()->where('type', TypeEnum::SUBSCRIPTION->value)
            ->where('active', 1)
            ->where(function ($query) {
                $query->where('price', 0)
                    ->orWhere('frequency', FrequencyEnum::LIFETIME_MONTHLY)
                    ->orWhere('frequency', FrequencyEnum::LIFETIME_YEARLY);
            })
            ->get(),
        '_subs_plans'
    );
}

function formatCamelCase($input)
{
    return ucwords(str_replace('_', ' ', $input));
}

function checkCouponInRequest($code = null)
{
    if ($code !== null) {
        $coupon = Coupon::where('code', $code)->where('is_offer', false)->first();
    } else {
        $couponCode = request()?->input('coupon');
        if ($couponCode) {
            $coupon = Coupon::where('code', $couponCode)->where('is_offer', false)->first();
        } else {
            $coupon = null;
        }
    }

    if (! $coupon) {
        $discountCoupon = null;
        if (\App\Helpers\Classes\MarketplaceHelper::isRegistered('discount-manager')) {
            $discountCoupon = \App\Extensions\DiscountManager\System\Services\DiscountService::applyDiscountCoupon();
        }

        if (! is_null($discountCoupon)) {
            return $discountCoupon;
        }
    }

    return $coupon;
}

function taxToVal($price, $tax)
{
    return ($tax > 0) ? ($price * $tax / 100) : 0;
}

function displayCurr($symbol, $price, $taxValue = 0, $discountedprice = null, $tax_included = false)
{
    $newPrice = $tax_included ? $price : $price + $taxValue;

    try {
        if (in_array($symbol, config('currency.currencies_with_right_symbols'))) {
            if ($discountedprice !== null && $discountedprice < $price) {
                return "<span style='text-decoration: line-through; color:grey;'>" . $newPrice . $symbol . '</span> <b>' . $discountedprice + $taxValue . $symbol . '</b>';
            }

            return $newPrice . $symbol;
        }

        if ($discountedprice !== null && $discountedprice < $price) {
            return "<span style='text-decoration: line-through; color:grey;'>" . $symbol . $newPrice . '</span> <b>' . $symbol . $discountedprice + $taxValue . '</b>';
        }

        return $symbol . $newPrice;
    } catch (\Exception $th) {
        $discountedprice = (float) str_replace(',', '', $discountedprice);
        $price = (float) str_replace(',', '', $price);
        if (in_array($symbol, config('currency.currencies_with_right_symbols'))) {
            if ($discountedprice !== null && $discountedprice < $price) {
                return "<span style='text-decoration: line-through; color:grey;'>" . $newPrice . $symbol . '</span> <b>' . $discountedprice + $taxValue . $symbol . '</b>';
            }

            return $newPrice . $symbol;
        }

        if ($discountedprice !== null && $discountedprice < $price) {
            return "<span style='text-decoration: line-through; color:grey;'>" . $symbol . $newPrice . '</span> <b>' . $symbol . $discountedprice + $taxValue . '</b>';
        }

        return $symbol . $newPrice;
    }

}

function displayCurrPlan($symbol, $price, $discountedprice = null)
{
    if (in_array($symbol, config('currency.currencies_with_right_symbols'))) {
        if ($discountedprice !== null && $discountedprice < $price) {
            return "<small class='inline-flex mb-[0.3em] font-normal text-[0.35em]'><span style='text-decoration: line-through;'>" . $price . '</span>' . $symbol . '</small>&nbsp;' . $discountedprice . "<small class='inline-flex mb-[0.3em] font-normal text-[0.35em]'>" . $symbol . '</small>';
        }

        return '<span>' . $price . "</span> <small class='inline-flex mb-[0.3em] font-normal text-[0.35em]'>" . $symbol . '</small>';
    }

    if ($discountedprice !== null && $discountedprice < $price) {
        return "<small class='inline-flex mb-[0.3em] font-normal text-[0.35em]'>" . $symbol . "<span style='text-decoration: line-through;'>" . $price . "</span> </small>&nbsp; <small class='inline-flex mb-[0.3em] font-normal text-[0.35em]'>" . $symbol . '</small>' . $discountedprice;
    }

    return "<small class='inline-flex mb-[0.3em] font-normal text-[0.35em]'>" . $symbol . '</small> <span>' . $price . '</span>';
}

function activeRoute(...$route_name): string
{

    $exceptListSlugs = [
        'ai_pdf', 'ai_vision', 'ai_chat_image', 'ai_code_generator', 'ai_youtube', 'ai_rss',
    ];

    $slug = request()?->route('slug');

    if ($slug && in_array($slug, $exceptListSlugs)) {
        return '';
    }

    return request()?->routeIs(...$route_name) ? 'active' : '';
}

function activeRouteBulk(...$route_names)
{
    return request()?->routeIs(...$route_names) ? 'active' : '';
}

function activeRouteBulkShow(...$route_names)
{
    return request()?->routeIs(...$route_names) ? 'show' : '';
}

if (! function_exists('custom_theme_url')) {
    function custom_theme_url($url, $slash = false)
    {
        if (strpos($url, 'assets') !== false) {
            return theme_url($url);
        }
        if ($slash) {
            return '/' . $url;
        }

        return $url;
    }
}

if (! function_exists('get_theme')) {
    function get_theme()
    {
        $theme = \Theme::get();
        if (! $theme) {
            $theme = 'default';
        }

        return $theme;
    }
}

if (! function_exists('check_plan_affiliate_status')) {
    function check_plan_affiliate_status($user): bool
    {
        $plan = $user?->activePlan();
        if ($plan) {
            return $plan->affiliate_status;
        }

        return false;
    }
}

// hsl_to_hex
if (! function_exists('hsl_to_hex')) {
    function hsl_to_hex($h, $s, $l)
    {
        $h /= 360;
        $s /= 100;
        $l /= 100;

        if ($s === 0) {
            $r = $g = $b = $l;
        } else {
            $hTemp = function ($p, $q, $t) {
                if ($t < 0) {
                    $t += 1;
                }
                if ($t > 1) {
                    $t -= 1;
                }
                if ($t < 1 / 6) {
                    return $p + ($q - $p) * 6 * $t;
                }
                if ($t < 1 / 2) {
                    return $q;
                }
                if ($t < 2 / 3) {
                    return $p + ($q - $p) * (2 / 3 - $t) * 6;
                }

                return $p;
            };

            $q = $l < 0.5 ? $l * (1 + $s) : $l + $s - $l * $s;
            $p = 2 * $l - $q;
            $r = $hTemp($p, $q, $h + 1 / 3);
            $g = $hTemp($p, $q, $h);
            $b = $hTemp($p, $q, $h - 1 / 3);
        }

        $r = round($r * 255);
        $g = round($g * 255);
        $b = round($b * 255);

        return sprintf('#%02x%02x%02x', $r, $g, $b);
    }
}
// hex_to_hsl
if (! function_exists('hex_to_hsl')) {
    function hex_to_hsl($hex)
    {
        // Remove '#' if present
        $hex = str_replace('#', '', $hex);

        // Convert hex to RGB
        $r = hexdec(substr($hex, 0, 2)) / 255;
        $g = hexdec(substr($hex, 2, 2)) / 255;
        $b = hexdec(substr($hex, 4, 2)) / 255;

        // Find the maximum and minimum values
        $max = max($r, $g, $b);
        $min = min($r, $g, $b);

        // Calculate the lightness
        $l = ($max + $min) / 2;

        if ($max === $min) {
            $h = $s = 0; // achromatic
        } else {
            $d = $max - $min;
            $s = $l > 0.5 ? $d / (2 - $max - $min) : $d / ($max + $min);

            switch ($max) {
                case $r:
                    $h = ($g - $b) / $d + ($g < $b ? 6 : 0);

                    break;
                case $g:
                    $h = ($b - $r) / $d + 2;

                    break;
                case $b:
                    $h = ($r - $g) / $d + 4;

                    break;
            }
            $h /= 6;
        }

        // Convert to percentage
        $h = round($h * 360);
        $s = round($s * 100);
        $l = round($l * 100);

        return "$h, $s%, $l%";
    }
}

function percentageChange($old, $new, int $precision = 1)
{
    if ($old == 0) {
        $old++;
        $new++;
    }
    $change = round((($new - $old) / $old) * 100, $precision);

    // Limiting the percentage change to be between 0 and 100
    $change = max(0, min(100, $change));

    return $change;
}

function percentageChangeSign($old, $new, int $precision = 2)
{

    if (percentageChange($old, $new) > 0) {
        return 'plus';
    }

    return 'minus';
}

function currency()
{
    $setting = \App\Models\Setting::getCache();

    $currency = \App\Models\Currency::cacheFromSetting($setting->default_currency);

    if (in_array($currency->code, config('currency.needs_code_with_symbol'), true)) {
        $currency->symbol = $currency->code . ' ' . $currency->symbol;
    }

    return $currency;
}

function getSubscription()
{
    $activeSub = getCurrentActiveSubscription();
    if ($activeSub == null) {
        $activeSub = getCurrentActiveSubscriptionYokkasa();
    }

    return $activeSub;
}

function getSubscriptionActive()
{
    return getSubscription();
}

function getSubscriptionStatus()
{
    return PaymentProcessController::getSubscriptionStatus();
}

function checkIfTrial()
{
    return PaymentProcessController::checkIfTrial();
}

function getSubscriptionName()
{
    $subscription = getSubscription();

    return $subscription !== null ?
        Plan::getCache(
            static fn () => Plan::query()->where('id', $subscription->plan_id)->first(),
            '-' . $subscription->plan_id
        )?->name
        : '';
}

function getSubscriptionRenewDate()
{
    return PaymentProcessController::getSubscriptionRenewDate();
}

function getSubscriptionDaysLeft()
{
    return PaymentProcessController::getSubscriptionDaysLeft();
}

// Templates favorited
function isFavorited($template_id)
{
    $isFav = \App\Models\UserFavorite::where('user_id', Auth::id())->where('openai_id', $template_id)->exists();

    return $isFav;
}

// Docs favorited
function isFavoritedDoc($template_id)
{
    $isFav = \App\Models\UserDocsFavorite::where('user_id', Auth::id())->where('user_openai_id', $template_id)->exists();

    return $isFav;
}

// Country Flags
function country2flag(string $countryCode): string
{

    if (strpos($countryCode, '-') !== false) {
        $countryCode = substr($countryCode, strpos($countryCode, '-') + 1);
    } elseif (strpos($countryCode, '_') !== false) {
        $countryCode = substr($countryCode, strpos($countryCode, '_') + 1);
    }

    if ($countryCode === 'el') {
        $countryCode = 'gr';
    } elseif ($countryCode === 'da') {
        $countryCode = 'dk';
    }

    return (string) preg_replace_callback(
        '/./',
        static fn (array $letter) => mb_chr(ord($letter[0]) % 32 + 0x1F1E5),
        $countryCode
    );
}

// Memory Limit
function getServerMemoryLimit()
{
    return (int) ini_get('memory_limit');
}

// Count Words
function countWords($text)
{

    $encoding = mb_detect_encoding($text);

    if ($encoding === 'UTF-8') {
        // Count Chinese words by splitting the string into individual characters
        $words = preg_match_all('/\p{Han}|\p{L}+|\p{N}+/u', $text);
    } else {
        // For other languages, use str_word_count()
        $words = str_word_count($text, 0, $encoding);
    }

    return (int) $words;

}

function getDefinedLangs()
{
    $fields = \DB::getSchemaBuilder()->getColumnListing('strings');
    $exceptions = ['en', 'code', 'created_at', 'updated_at'];
    $filtered = collect($fields)->filter(function ($value, $key) use ($exceptions) {
        if (! in_array($value, $exceptions)) {
            return $value;
        }
    });

    return $filtered->all();
}

function getVoiceNames($hash)
{
    $voiceNames = [
        'af-ZA-Standard-A'  => 'Ayanda (' . __('Female') . ')',
        'ar-XA-Standard-A'  => 'Fatima (' . __('Female') . ')',
        'ar-XA-Standard-B'  => 'Ahmed (' . __('Male') . ')',
        'ar-XA-Standard-C'  => 'Mohammed (' . __('Male') . ')',
        'ar-XA-Standard-D'  => 'Aisha (' . __('Female') . ')',
        'ar-XA-Wavenet-A'   => 'Layla (' . __('Female') . ')',
        'ar-XA-Wavenet-B'   => 'Ali (' . __('Male') . ')',
        'ar-XA-Wavenet-C'   => 'Omar (' . __('Male') . ')',
        'ar-XA-Wavenet-D'   => 'Zahra (' . __('Female') . ')',
        'eu-ES-Standard-A'  => 'Ane (' . __('Female') . ')',
        'bn-IN-Standard-A'  => 'Ananya (' . __('Female') . ')',
        'bn-IN-Standard-B'  => 'Aryan (' . __('Male') . ')',
        'bn-IN-Wavenet-A'   => 'Ishita (' . __('Female') . ')',
        'bn-IN-Wavenet-B'   => 'Arry (' . __('Male') . ')',
        'bg-BG-Standard-A'  => 'Elena (' . __('Female') . ')',
        'ca-ES-Standard-A'  => 'Laia (' . __('Female') . ')',
        'yue-HK-Standard-A' => 'Wing (' . __('Female') . ')',
        'yue-HK-Standard-B' => 'Ho (' . __('Male') . ')',
        'yue-HK-Standard-C' => 'Siu (' . __('Female') . ')',
        'yue-HK-Standard-D' => 'Lau (' . __('Male') . ')',
        'cs-CZ-Standard-A'  => 'Tereza (' . __('Female') . ')',
        'cs-CZ-Wavenet-A'   => 'Karolína (' . __('Female') . ')',
        // "da-DK-Neural2-D" => "Neural2 - FEMALE",
        // "da-DK-Neural2-F" => "Neural2 - MALE",
        'da-DK-Standard-A' => 'Emma (' . __('Female') . ')',
        // 'da-DK-Standard-A' => 'Freja (' . __('Female') . ')',
        // 'da-DK-Standard-A' => 'Ida (' . __('Female') . ')',
        'da-DK-Standard-C' => 'Noah (' . __('Male') . ')',
        'da-DK-Standard-D' => 'Mathilde (' . __('Female') . ')',
        'da-DK-Standard-E' => 'Clara (' . __('Female') . ')',
        'da-DK-Wavenet-A'  => 'Isabella (' . __('Female') . ')',
        'da-DK-Wavenet-C'  => 'Lucas (' . __('Male') . ')',
        'da-DK-Wavenet-D'  => 'Olivia (' . __('Female') . ')',
        'da-DK-Wavenet-E'  => 'Emily (' . __('Female') . ')',
        'nl-BE-Standard-A' => 'Emma (' . __('Female') . ')',
        'nl-BE-Standard-B' => 'Thomas (' . __('Male') . ')',
        'nl-BE-Wavenet-A'  => 'Sophie (' . __('Female') . ')',
        'nl-BE-Wavenet-B'  => 'Lucas (' . __('Male') . ')',
        'nl-NL-Standard-A' => 'Emma (' . __('Female') . ')',
        'nl-NL-Standard-B' => 'Daan (' . __('Male') . ')',
        'nl-NL-Standard-C' => 'Luuk (' . __('Male') . ')',
        'nl-NL-Standard-D' => 'Lotte (' . __('Female') . ')',
        'nl-NL-Standard-E' => 'Sophie (' . __('Female') . ')',
        'nl-NL-Wavenet-A'  => 'Mila (' . __('Female') . ')',
        'nl-NL-Wavenet-B'  => 'Sem (' . __('Male') . ')',
        'nl-NL-Wavenet-C'  => 'Stijn (' . __('Male') . ')',
        'nl-NL-Wavenet-D'  => 'Fenna (' . __('Female') . ')',
        'nl-NL-Wavenet-E'  => 'Eva (' . __('Female') . ')',
        // "en-AU-Neural2-A" => "Neural2 - FEMALE",
        // "en-AU-Neural2-B" => "Neural2 - MALE",
        // "en-AU-Neural2-C" => "Neural2 - FEMALE",
        // "en-AU-Neural2-D" => "Neural2 - MALE",
        'en-AU-News-E'     => 'Emma (' . __('Female') . ')',
        'en-AU-News-F'     => 'Olivia (' . __('Female') . ')',
        'en-AU-News-G'     => 'Liam (' . __('Male') . ')',
        'en-AU-Polyglot-1' => 'Noah (' . __('Male') . ')',
        'en-AU-Standard-A' => 'Charlotte (' . __('Female') . ')',
        'en-AU-Standard-B' => 'Oliver (' . __('Male') . ')',
        'en-AU-Standard-C' => 'Ava (' . __('Female') . ')',
        'en-AU-Standard-D' => 'Jack (' . __('Male') . ')',
        'en-AU-Wavenet-A'  => 'Sophie (' . __('Female') . ')',
        'en-AU-Wavenet-B'  => 'William (' . __('Male') . ')',
        'en-AU-Wavenet-C'  => 'Amelia (' . __('Female') . ')',
        'en-AU-Wavenet-D'  => 'Thomas (' . __('Male') . ')',
        'en-IN-Standard-A' => 'Aditi (' . __('Female') . ')',
        'en-IN-Standard-B' => 'Arjun (' . __('Male') . ')',
        'en-IN-Standard-C' => 'Rohan (' . __('Male') . ')',
        'en-IN-Standard-D' => 'Ananya (' . __('Female') . ')',
        'en-IN-Wavenet-A'  => 'Alisha (' . __('Female') . ')',
        'en-IN-Wavenet-B'  => 'Aryan (' . __('Male') . ')',
        'en-IN-Wavenet-C'  => 'Kabir (' . __('Male') . ')',
        'en-IN-Wavenet-D'  => 'Diya (' . __('Female') . ')',
        // "en-GB-Neural2-A" => "Neural2 - FEMALE",
        // "en-GB-Neural2-B" => "Neural2 - MALE",
        // "en-GB-Neural2-C" => "Neural2 - FEMALE",
        // "en-GB-Neural2-D" => "Neural2 - MALE",
        // "en-GB-Neural2-F" => "Neural2 - FEMALE",
        'en-GB-News-G'     => 'Amelia (' . __('Female') . ')',
        'en-GB-News-H'     => 'Elise (' . __('Female') . ')',
        'en-GB-News-I'     => 'Isabella (' . __('Female') . ')',
        'en-GB-News-J'     => 'Jessica (' . __('Female') . ')',
        'en-GB-News-K'     => 'Alexander (' . __('Male') . ')',
        'en-GB-News-L'     => 'Benjamin (' . __('Male') . ')',
        'en-GB-News-M'     => 'Charles (' . __('Male') . ')',
        'en-GB-Standard-A' => 'Emily (' . __('Female') . ')',
        'en-GB-Standard-B' => 'John (' . __('Male') . ')',
        'en-GB-Standard-C' => 'Mary (' . __('Female') . ')',
        'en-GB-Standard-D' => 'Peter (' . __('Male') . ')',
        'en-GB-Standard-F' => 'Sarah (' . __('Female') . ')',
        'en-GB-Wavenet-A'  => 'Ava (' . __('Female') . ')',
        'en-GB-Wavenet-B'  => 'David (' . __('Male') . ')',
        'en-GB-Wavenet-C'  => 'Emily (' . __('Female') . ')',
        'en-GB-Wavenet-D'  => 'James (' . __('Male') . ')',
        'en-GB-Wavenet-F'  => 'Sophie (' . __('Female') . ')',
        // "en-US-Neural2-A" => "Neural2 - MALE",
        // "en-US-Neural2-C" => "Neural2 - FEMALE",
        // "en-US-Neural2-D" => "Neural2 - MALE",
        // "en-US-Neural2-E" => "Neural2 - FEMALE",
        // "en-US-Neural2-F" => "Neural2 - FEMALE",
        // "en-US-Neural2-G" => "Neural2 - FEMALE",
        // "en-US-Neural2-H" => "Neural2 - FEMALE",
        // "en-US-Neural2-I" => "Neural2 - MALE",
        // "en-US-Neural2-J" => "Neural2 - MALE",
        'en-US-News-K'      => 'Lily (' . __('Female') . ')',
        'en-US-News-L'      => 'Olivia (' . __('Female') . ')',
        'en-US-News-M'      => 'Noah (' . __('Male') . ')',
        'en-US-News-N'      => 'Oliver (' . __('Male') . ')',
        'en-US-Polyglot-1'  => 'John (' . __('Male') . ')',
        'en-US-Standard-A'  => 'Michael (' . __('Male') . ')',
        'en-US-Standard-B'  => 'David (' . __('Male') . ')',
        'en-US-Standard-C'  => 'Emma (' . __('Female') . ')',
        'en-US-Standard-D'  => 'William (' . __('Male') . ')',
        'en-US-Standard-E'  => 'Ava (' . __('Female') . ')',
        'en-US-Standard-F'  => 'Sophia (' . __('Female') . ')',
        'en-US-Standard-G'  => 'Isabella (' . __('Female') . ')',
        'en-US-Standard-H'  => 'Charlotte (' . __('Female') . ')',
        'en-US-Standard-I'  => 'James (' . __('Male') . ')',
        'en-US-Standard-J'  => 'Lucas (' . __('Male') . ')',
        'en-US-Studio-M'    => 'Benjamin (' . __('Male') . ')',
        'en-US-Studio-O'    => 'Eleanor (' . __('Female') . ')',
        'en-US-Wavenet-A'   => 'Alexander (' . __('Male') . ')',
        'en-US-Wavenet-B'   => 'Benjamin (' . __('Male') . ')',
        'en-US-Wavenet-C'   => 'Emily (' . __('Female') . ')',
        'en-US-Wavenet-D'   => 'James (' . __('Male') . ')',
        'en-US-Wavenet-E'   => 'Ava (' . __('Female') . ')',
        'en-US-Wavenet-F'   => 'Sophia (' . __('Female') . ')',
        'en-US-Wavenet-G'   => 'Isabella (' . __('Female') . ')',
        'en-US-Wavenet-H'   => 'Charlotte (' . __('Female') . ')',
        'en-US-Wavenet-I'   => 'Alexander (' . __('Male') . ')',
        'en-US-Wavenet-J'   => 'Lucas (' . __('Male') . ')',
        'fil-PH-Standard-A' => 'Maria (' . __('Female') . ')',
        'fil-PH-Standard-B' => 'Juana (' . __('Female') . ')',
        'fil-PH-Standard-C' => 'Juan (' . __('Male') . ')',
        'fil-PH-Standard-D' => 'Pedro (' . __('Male') . ')',
        'fil-PH-Wavenet-A'  => 'Maria (' . __('Female') . ')',
        'fil-PH-Wavenet-B'  => 'Juana (' . __('Female') . ')',
        'fil-PH-Wavenet-C'  => 'Juan (' . __('Male') . ')',
        'fil-PH-Wavenet-D'  => 'Pedro (' . __('Male') . ')',
        // "fil-ph-Neural2-A" => "Neural2 - FEMALE",
        // "fil-ph-Neural2-D" => "Neural2 - MALE",
        'fi-FI-Standard-A' => 'Sofia (' . __('Female') . ')',
        'fi-FI-Wavenet-A'  => 'Sofianna (' . __('Female') . ')',
        // "fr-CA-Neural2-A" => "Neural2 - FEMALE",
        // "fr-CA-Neural2-B" => "Neural2 - MALE",
        // "fr-CA-Neural2-C" => "Neural2 - FEMALE",
        // "fr-CA-Neural2-D" => "Neural2 - MALE",
        'fr-CA-Standard-A' => 'Emma (' . __('Female') . ')',
        'fr-CA-Standard-B' => 'Jean (' . __('Male') . ')',
        'fr-CA-Standard-C' => 'Gabrielle (' . __('Female') . ')',
        'fr-CA-Standard-D' => 'Thomas (' . __('Male') . ')',
        'fr-CA-Wavenet-A'  => 'Amelie (' . __('Female') . ')',
        'fr-CA-Wavenet-B'  => 'Antoine (' . __('Male') . ')',
        'fr-CA-Wavenet-C'  => 'Gabrielle (' . __('Female') . ')',
        'fr-CA-Wavenet-D'  => 'Thomas (' . __('Male') . ')',
        // "fr-FR-Neural2-A" => "Neural2 - FEMALE",
        // "fr-FR-Neural2-B" => "Neural2 - MALE",
        // "fr-FR-Neural2-C" => "Neural2 - FEMALE",
        // "fr-FR-Neural2-D" => "Neural2 - MALE",
        // "fr-FR-Neural2-E" => "Neural2 - FEMALE",
        'fr-FR-Polyglot-1' => 'Jean (' . __('Male') . ')',
        'fr-FR-Standard-A' => 'Marie (' . __('Female') . ')',
        'fr-FR-Standard-B' => 'Pierre (' . __('Male') . ')',
        'fr-FR-Standard-C' => 'Sophie (' . __('Female') . ')',
        'fr-FR-Standard-D' => 'Paul (' . __('Male') . ')',
        'fr-FR-Standard-E' => 'Julie (' . __('Female') . ')',
        'fr-FR-Wavenet-A'  => 'Elise (' . __('Female') . ')',
        'fr-FR-Wavenet-B'  => 'Nicolas (' . __('Male') . ')',
        'fr-FR-Wavenet-C'  => 'Clara (' . __('Female') . ')',
        'fr-FR-Wavenet-D'  => 'Antoine (' . __('Male') . ')',
        'fr-FR-Wavenet-E'  => 'Amelie (' . __('Female') . ')',
        'gl-ES-Standard-A' => 'Ana (' . __('Female') . ')',
        // "de-DE-Neural2-B" => "Neural2 - MALE",
        // "de-DE-Neural2-C" => "Neural2 - FEMALE",
        // "de-DE-Neural2-D" => "Neural2 - MALE",
        // "de-DE-Neural2-F" => "Neural2 - FEMALE",
        'de-DE-Polyglot-1' => 'Johannes (' . __('Male') . ')',
        'de-DE-Standard-A' => 'Anna (' . __('Female') . ')',
        'de-DE-Standard-B' => 'Max (' . __('Male') . ')',
        'de-DE-Standard-C' => 'Sophia (' . __('Female') . ')',
        'de-DE-Standard-D' => 'Paul (' . __('Male') . ')',
        'de-DE-Standard-E' => 'Erik (' . __('Male') . ')',
        'de-DE-Standard-F' => 'Lina (' . __('Female') . ')',
        'de-DE-Wavenet-A'  => 'Eva (' . __('Female') . ')',
        'de-DE-Wavenet-B'  => 'Felix (' . __('Male') . ')',
        'de-DE-Wavenet-C'  => 'Emma (' . __('Female') . ')',
        'de-DE-Wavenet-D'  => 'Lukas (' . __('Male') . ')',
        'de-DE-Wavenet-E'  => 'Nico (' . __('Male') . ')',
        'de-DE-Wavenet-F'  => 'Mia (' . __('Female') . ')',
        'el-GR-Standard-A' => 'Ελένη (' . __('Female') . ')',
        'el-GR-Wavenet-A'  => 'Ελένη (' . __('Female') . ')',
        'gu-IN-Standard-A' => 'દિવ્યા (' . __('Female') . ')',
        'gu-IN-Standard-B' => 'કિશોર (' . __('Male') . ')',
        'gu-IN-Wavenet-A'  => 'દિવ્યા (' . __('Female') . ')',
        'gu-IN-Wavenet-B'  => 'કિશોર (' . __('Male') . ')',
        'he-IL-Standard-A' => 'Tamar (' . __('Female') . ')',
        'he-IL-Standard-B' => 'David (' . __('Male') . ')',
        'he-IL-Standard-C' => 'Michal (' . __('Female') . ')',
        'he-IL-Standard-D' => 'Jonathan (' . __('Male') . ')',
        'he-IL-Wavenet-A'  => 'Yael (' . __('Female') . ')',
        'he-IL-Wavenet-B'  => 'Eli (' . __('Male') . ')',
        'he-IL-Wavenet-C'  => 'Abigail (' . __('Female') . ')',
        'he-IL-Wavenet-D'  => 'Alex (' . __('Male') . ')',
        // "hi-IN-Neural2-A" => "Neural2 - FEMALE",
        // "hi-IN-Neural2-B" => "Neural2 - MALE",
        // "hi-IN-Neural2-C" => "Neural2 - MALE",
        // "hi-IN-Neural2-D" => "Neural2 - FEMALE",
        'hi-IN-Standard-A' => 'Aditi (' . __('Female') . ')',
        'hi-IN-Standard-B' => 'Abhishek (' . __('Male') . ')',
        'hi-IN-Standard-C' => 'Aditya (' . __('Male') . ')',
        'hi-IN-Standard-D' => 'Anjali (' . __('Female') . ')',
        'hi-IN-Wavenet-A'  => 'Kiara (' . __('Female') . ')',
        'hi-IN-Wavenet-B'  => 'Rohan (' . __('Male') . ')',
        'hi-IN-Wavenet-C'  => 'Rishabh (' . __('Male') . ')',
        'hi-IN-Wavenet-D'  => 'Srishti (' . __('Female') . ')',
        'hu-HU-Standard-A' => 'Eszter (' . __('Female') . ')',
        'hu-HU-Wavenet-A'  => 'Lilla (' . __('Female') . ')',
        'is-IS-Standard-A' => 'Guðrún (' . __('Female') . ')',
        'id-ID-Standard-A' => 'Amelia (' . __('Female') . ')',
        'id-ID-Standard-B' => 'Fajar (' . __('Male') . ')',
        'id-ID-Standard-C' => 'Galih (' . __('Male') . ')',
        'id-ID-Standard-D' => 'Kiara (' . __('Female') . ')',
        'id-ID-Wavenet-A'  => 'Nadia (' . __('Female') . ')',
        'id-ID-Wavenet-B'  => 'Reza (' . __('Male') . ')',
        'id-ID-Wavenet-C'  => 'Satria (' . __('Male') . ')',
        'id-ID-Wavenet-D'  => 'Vania (' . __('Female') . ')',
        // "it-IT-Neural2-A" => "Neural2 - FEMALE",
        // "it-IT-Neural2-C" => "Neural2 - MALE",
        'it-IT-Standard-A' => 'Chiara (' . __('Female') . ')',
        'it-IT-Standard-B' => 'Elisa (' . __('Female') . ')',
        'it-IT-Standard-C' => 'Matteo (' . __('Male') . ')',
        'it-IT-Standard-D' => 'Riccardo (' . __('Male') . ')',
        'it-IT-Wavenet-A'  => 'Valentina (' . __('Female') . ')',
        'it-IT-Wavenet-B'  => 'Vittoria (' . __('Female') . ')',
        'it-IT-Wavenet-C'  => 'Andrea (' . __('Male') . ')',
        'it-IT-Wavenet-D'  => 'Luca (' . __('Male') . ')',
        // "ja-JP-Neural2-B" => "Neural2 - FEMALE",
        // "ja-JP-Neural2-C" => "Neural2 - MALE",
        // "ja-JP-Neural2-D" => "Neural2 - MALE",
        'ja-JP-Standard-A' => 'Akane (' . __('Female') . ')',
        'ja-JP-Standard-B' => 'Emi (' . __('Female') . ')',
        'ja-JP-Standard-C' => 'Daisuke (' . __('Male') . ')',
        'ja-JP-Standard-D' => 'Kento (' . __('Male') . ')',
        'ja-JP-Wavenet-A'  => 'Haruka (' . __('Female') . ')',
        'ja-JP-Wavenet-B'  => 'Rin (' . __('Female') . ')',
        'ja-JP-Wavenet-C'  => 'Shun (' . __('Male') . ')',
        'ja-JP-Wavenet-D'  => 'Yuta (' . __('Male') . ')',
        'kn-IN-Standard-A' => 'Dhanya (' . __('Female') . ')',
        'kn-IN-Standard-B' => 'Keerthi (' . __('Male') . ')',
        'kn-IN-Wavenet-A'  => 'Meena (' . __('Female') . ')',
        'kn-IN-Wavenet-B'  => 'Nandini (' . __('Male') . ')',
        // "ko-KR-Neural2-A" => "Neural2 - FEMALE",
        // "ko-KR-Neural2-B" => "Neural2 - FEMALE",
        // "ko-KR-Neural2-C" => "Neural2 - MALE",
        'ko-KR-Standard-A'  => 'So-young (' . __('Female') . ')',
        'ko-KR-Standard-B'  => 'Se-yeon (' . __('Female') . ')',
        'ko-KR-Standard-C'  => 'Min-soo (' . __('Male') . ')',
        'ko-KR-Standard-D'  => 'Seung-woo (' . __('Male') . ')',
        'ko-KR-Wavenet-A'   => 'Ji-soo (' . __('Female') . ')',
        'ko-KR-Wavenet-B'   => 'Yoon-a (' . __('Female') . ')',
        'ko-KR-Wavenet-C'   => 'Tae-hyun (' . __('Male') . ')',
        'ko-KR-Wavenet-D'   => 'Jun-ho (' . __('Male') . ')',
        'lv-LV-Standard-A'  => 'Raivis (' . __('Male') . ')',
        'lv-LT-Standard-A'  => 'Raivis (' . __('Male') . ')',
        'ms-MY-Standard-A'  => 'Amira (' . __('Female') . ')',
        'ms-MY-Standard-B'  => 'Danial (' . __('Male') . ')',
        'ms-MY-Standard-C'  => 'Eira (' . __('Female') . ')',
        'ms-MY-Standard-D'  => 'Farhan (' . __('Male') . ')',
        'ms-MY-Wavenet-A'   => 'Hana (' . __('Female') . ')',
        'ms-MY-Wavenet-B'   => 'Irfan (' . __('Male') . ')',
        'ms-MY-Wavenet-C'   => 'Janna (' . __('Female') . ')',
        'ms-MY-Wavenet-D'   => 'Khairul (' . __('Male') . ')',
        'ml-IN-Standard-A'  => 'Aishwarya (' . __('Female') . ')',
        'ml-IN-Standard-B'  => 'Dhruv (' . __('Male') . ')',
        'ml-IN-Wavenet-A'   => 'Deepthi (' . __('Female') . ')',
        'ml-IN-Wavenet-B'   => 'Gautam (' . __('Male') . ')',
        'ml-IN-Wavenet-C'   => 'Isha (' . __('Female') . ')',
        'ml-IN-Wavenet-D'   => 'Kabir (' . __('Male') . ')',
        'cmn-CN-Standard-A' => 'Xiaomei (' . __('Female') . ')',
        'cmn-CN-Standard-B' => 'Lijun (' . __('Male') . ')',
        'cmn-CN-Standard-C' => 'Minghao (' . __('Male') . ')',
        'cmn-CN-Standard-D' => 'Yingying (' . __('Female') . ')',
        'cmn-CN-Wavenet-A'  => 'Shanshan (' . __('Female') . ')',
        'cmn-CN-Wavenet-B'  => 'Chenchen (' . __('Male') . ')',
        'cmn-CN-Wavenet-C'  => 'Jiahao (' . __('Male') . ')',
        'cmn-CN-Wavenet-D'  => 'Yueyu (' . __('Female') . ')',
        'cmn-TW-Standard-A' => 'Jingwen (' . __('Female') . ')',
        'cmn-TW-Standard-B' => 'Jinghao (' . __('Male') . ')',
        'cmn-TW-Standard-C' => 'Tingting (' . __('Female') . ')',
        'cmn-TW-Wavenet-A'  => 'Yunyun (' . __('Female') . ')',
        'cmn-TW-Wavenet-B'  => 'Zhenghao (' . __('Male') . ')',
        'cmn-TW-Wavenet-C'  => 'Yuehan (' . __('Female') . ')',
        'mr-IN-Standard-A'  => 'Anjali (' . __('Female') . ')',
        'mr-IN-Standard-B'  => 'Aditya (' . __('Male') . ')',
        'mr-IN-Standard-C'  => 'Dipti (' . __('Female') . ')',
        'mr-IN-Wavenet-A'   => 'Gauri (' . __('Female') . ')',
        'mr-IN-Wavenet-B'   => 'Harsh (' . __('Male') . ')',
        'mr-IN-Wavenet-C'   => 'Ishita (' . __('Female') . ')',
        'nb-NO-Standard-A'  => 'Ingrid (' . __('Female') . ')',
        'nb-NO-Standard-B'  => 'Jonas (' . __('Male') . ')',
        'nb-NO-Standard-C'  => 'Marit (' . __('Female') . ')',
        'nb-NO-Standard-D'  => 'Olav (' . __('Male') . ')',
        'nb-NO-Standard-E'  => 'Silje (' . __('Female') . ')',
        'nb-NO-Wavenet-A'   => 'Astrid (' . __('Female') . ')',
        'nb-NO-Wavenet-B'   => 'Eirik (' . __('Male') . ')',
        'nb-NO-Wavenet-C'   => 'Inger (' . __('Female') . ')',
        'nb-NO-Wavenet-D'   => 'Kristian (' . __('Male') . ')',
        'nb-NO-Wavenet-E'   => 'Trine (' . __('Female') . ')',
        'pl-PL-Standard-A'  => 'Agata (' . __('Female') . ')',
        'pl-PL-Standard-B'  => 'Bartosz (' . __('Male') . ')',
        'pl-PL-Standard-C'  => 'Kamil (' . __('Male') . ')',
        'pl-PL-Standard-D'  => 'Julia (' . __('Female') . ')',
        'pl-PL-Standard-E'  => 'Magdalena (' . __('Female') . ')',
        'pl-PL-Wavenet-A'   => 'Natalia (' . __('Female') . ')',
        'pl-PL-Wavenet-B'   => 'Paweł (' . __('Male') . ')',
        'pl-PL-Wavenet-C'   => 'Tomasz (' . __('Male') . ')',
        'pl-PL-Wavenet-D'   => 'Zofia (' . __('Female') . ')',
        'pl-PL-Wavenet-E'   => 'Wiktoria (' . __('Female') . ')',
        // "pt-BR-Neural2-A" => "Neural2 - FEMALE",
        // "pt-BR-Neural2-B" => "Neural2 - MALE",
        // "pt-BR-Neural2-C" => "Neural2 - FEMALE",
        'pt-BR-Standard-A' => 'Ana (' . __('Female') . ')',
        'pt-BR-Standard-B' => 'Carlos (' . __('Male') . ')',
        'pt-BR-Standard-C' => 'Maria (' . __('Female') . ')',
        'pt-BR-Wavenet-A'  => 'Julia (' . __('Female') . ')',
        'pt-BR-Wavenet-B'  => 'João (' . __('Male') . ')',
        'pt-BR-Wavenet-C'  => 'Fernanda (' . __('Female') . ')',
        'pt-PT-Standard-A' => 'Maria (' . __('Female') . ')',
        'pt-PT-Standard-B' => 'José (' . __('Male') . ')',
        'pt-PT-Standard-C' => 'Luís (' . __('Male') . ')',
        'pt-PT-Standard-D' => 'Ana (' . __('Female') . ')',
        'pt-PT-Wavenet-A'  => 'Catarina (' . __('Female') . ')',
        'pt-PT-Wavenet-B'  => 'Miguel (' . __('Male') . ')',
        'pt-PT-Wavenet-C'  => 'João (' . __('Male') . ')',
        'pt-PT-Wavenet-D'  => 'Marta (' . __('Female') . ')',
        'pa-IN-Standard-A' => 'Harpreet (' . __('Female') . ')',
        'pa-IN-Standard-B' => 'Gurpreet (' . __('Male') . ')',
        'pa-IN-Standard-C' => 'Jasmine (' . __('Female') . ')',
        'pa-IN-Standard-D' => 'Rahul (' . __('Male') . ')',
        'pa-IN-Wavenet-A'  => 'Simran (' . __('Female') . ')',
        'pa-IN-Wavenet-B'  => 'Amardeep (' . __('Male') . ')',
        'pa-IN-Wavenet-C'  => 'Kiran (' . __('Female') . ')',
        'pa-IN-Wavenet-D'  => 'Raj (' . __('Male') . ')',
        'ro-RO-Standard-A' => 'Maria (' . __('Female') . ')',
        'ro-RO-Wavenet-A'  => 'Ioana (' . __('Female') . ')',
        'ru-RU-Standard-A' => 'Anastasia',
        'ru-RU-Standard-B' => 'Alexander',
        'ru-RU-Standard-C' => 'Elizabeth',
        'ru-RU-Standard-D' => 'Michael',
        'ru-RU-Standard-E' => 'Victoria',
        'ru-RU-Wavenet-A'  => 'Daria',
        'ru-RU-Wavenet-B'  => 'Dmitry',
        'ru-RU-Wavenet-C'  => 'Kristina',
        'ru-RU-Wavenet-D'  => 'Ivan',
        'ru-RU-Wavenet-E'  => 'Sophia',
        'sr-RS-Standard-A' => 'Ana',
        'sk-SK-Standard-A' => 'Mária (' . __('Female') . ')',
        'sk-SK-Wavenet-A'  => 'Zuzana (' . __('Female') . ')',
        // "es-ES-Neural2-A" => "Neural2 - FEMALE",
        // "es-ES-Neural2-B" => "Neural2 - MALE",
        // "es-ES-Neural2-C" => "Neural2 - FEMALE",
        // "es-ES-Neural2-D" => "Neural2 - FEMALE",
        // "es-ES-Neural2-E" => "Neural2 - FEMALE",
        // "es-ES-Neural2-F" => "Neural2 - MALE",
        'es-ES-Polyglot-1' => 'Juan (' . __('Male') . ')',
        'es-ES-Standard-A' => 'María (' . __('Female') . ')',
        'es-ES-Standard-B' => 'José (' . __('Male') . ')',
        'es-ES-Standard-C' => 'Ana (' . __('Female') . ')',
        'es-ES-Standard-D' => 'Isabel (' . __('Female') . ')',
        'es-ES-Wavenet-B'  => 'Pedro (' . __('Male') . ')',
        'es-ES-Wavenet-C'  => 'Laura (' . __('Female') . ')',
        'es-ES-Wavenet-D'  => 'Julia (' . __('Female') . ')',
        // "es-US-Neural2-A" => "Neural2 - FEMALE",
        // "es-US-Neural2-B" => "Neural2 - MALE",
        // "es-US-Neural2-C" => "Neural2 - MALE",
        'es-US-News-D'     => 'Diego (' . __('Male') . ')',
        'es-US-News-E'     => 'Eduardo (' . __('Male') . ')',
        'es-US-News-F'     => 'Fátima (' . __('Female') . ')',
        'es-US-News-G'     => 'Gabriela (' . __('Female') . ')',
        'es-US-Polyglot-1' => 'Juan (' . __('Male') . ')',
        'es-US-Standard-A' => 'Ana (' . __('Female') . ')',
        'es-US-Standard-B' => 'José (' . __('Male') . ')',
        'es-US-Standard-C' => 'Carlos (' . __('Male') . ')',
        'es-US-Studio-B'   => 'Miguel (' . __('Male') . ')',
        'es-US-Wavenet-A'  => 'Laura (' . __('Female') . ')',
        'es-US-Wavenet-B'  => 'Pedro (' . __('Male') . ')',
        'es-US-Wavenet-C'  => 'Pablo (' . __('Male') . ')',
        'sv-SE-Standard-A' => 'Ebba (' . __('Female') . ')',
        'sv-SE-Standard-B' => 'Saga (' . __('Female') . ')',
        'sv-SE-Standard-C' => 'Linnea (' . __('Female') . ')',
        'sv-SE-Standard-D' => 'Erik (' . __('Male') . ')',
        'sv-SE-Standard-E' => 'Anton (' . __('Male') . ')',
        'sv-SE-Wavenet-A'  => 'Astrid (' . __('Female') . ')',
        'sv-SE-Wavenet-B'  => 'Elin (' . __('Female') . ')',
        'sv-SE-Wavenet-C'  => 'Oskar (' . __('Male') . ')',
        'sv-SE-Wavenet-D'  => 'Hanna (' . __('Female') . ')',
        'sv-SE-Wavenet-E'  => 'Felix (' . __('Male') . ')',
        'ta-IN-Standard-A' => 'Anjali (' . __('Female') . ')',
        'ta-IN-Standard-B' => 'Karthik (' . __('Male') . ')',
        'ta-IN-Standard-C' => 'Priya (' . __('Female') . ')',
        'ta-IN-Standard-D' => 'Ravi (' . __('Male') . ')',
        'ta-IN-Wavenet-A'  => 'Lakshmi (' . __('Female') . ')',
        'ta-IN-Wavenet-B'  => 'Suresh (' . __('Male') . ')',
        'ta-IN-Wavenet-C'  => 'Uma (' . __('Female') . ')',
        'ta-IN-Wavenet-D'  => 'Venkatesh (' . __('Male') . ')',
        '-IN-Standard-A'   => 'Anjali - (' . __('Female') . ')',
        '-IN-Standard-B'   => 'Karthik - (' . __('Male') . ')',
        // "th-TH-Neural2-C" => "Neural2 - FEMALE",
        'th-TH-Standard-A' => 'Ariya - (' . __('Female') . ')',
        'tr-TR-Standard-A' => 'Ayşe (' . __('Female') . ')',
        'tr-TR-Standard-B' => 'Berk (' . __('Male') . ')',
        'tr-TR-Standard-C' => 'Cansu (' . __('Female') . ')',
        'tr-TR-Standard-D' => 'Deniz (' . __('Female') . ')',
        'tr-TR-Standard-E' => 'Emre (' . __('Male') . ')',
        'tr-TR-Wavenet-A'  => 'Gül (' . __('Female') . ')',
        'tr-TR-Wavenet-B'  => 'Mert (' . __('Male') . ')',
        'tr-TR-Wavenet-C'  => 'Nilay (' . __('Female') . ')',
        'tr-TR-Wavenet-D'  => 'Selin (' . __('Female') . ')',
        'tr-TR-Wavenet-E'  => 'Tolga (' . __('Male') . ')',
        'uk-UA-Standard-A' => 'Anya - (' . __('Female') . ')',
        'uk-UA-Wavenet-A'  => 'Dasha - (' . __('Female') . ')',
        // "vi-VN-Neural2-A" => "Neural2 - FEMALE",
        // "vi-VN-Neural2-D" => "Neural2 - MALE",
        'vi-VN-Standard-A' => 'Mai (' . __('Female') . ')',
        'vi-VN-Standard-B' => 'Nam (' . __('Male') . ')',
        'vi-VN-Standard-C' => 'Hoa (' . __('Female') . ')',
        'vi-VN-Standard-D' => 'Huy (' . __('Male') . ')',
        'vi-VN-Wavenet-A'  => 'Lan (' . __('Female') . ')',
        'vi-VN-Wavenet-B'  => 'Son (' . __('Male') . ')',
        'vi-VN-Wavenet-C'  => 'Thao (' . __('Female') . ')',
        'vi-VN-Wavenet-D'  => 'Tuan (' . __('Male') . ')',
    ];

    return $voiceNames[$hash] ?? $hash;
}

function format_double($number)
{
    $parts = explode('.', $number);

    if (count($parts) == 1) {
        return $parts[0] . '.0';
    }

    $integerPart = $parts[0];
    $decimalPart = isset($parts[1]) ? $parts[1] : '';

    if (strlen($decimalPart) > 1) {
        $secondDecimalPart = substr($decimalPart, 1);
    } else {
        $secondDecimalPart = '0';
    }

    return $integerPart . '.' . $decimalPart[0] . '.' . $secondDecimalPart;
}

function currencyShouldDisplayOnRight($currencySymbol)
{
    return in_array($currencySymbol, config('currency.currencies_with_right_symbols'), true);
}

function displayPlanPrice(?Plan $plan, ?Currency $currency): string
{
    static $cache = [];

    if (! $plan) {
        return '';
    }

    $planId = $plan->id ?? 'null';
    $currencySymbol = $currency?->symbol ?? '';
    $cacheKey = "{$planId}_{$currencySymbol}";

    if (isset($cache[$cacheKey])) {
        return $cache[$cacheKey];
    }

    $originalPrice = $plan->price;
    $discountedPrice = newPlanDiscountPrice($plan);
    $symbolOnRight = currencyShouldDisplayOnRight($currencySymbol);

    $format = static function ($amount) {
        return fmod($amount, 1) === 0.0
            ? number_format($amount, 0)
            : number_format($amount, 2);
    };

    $formattedOriginal = $format($originalPrice);
    $formattedDiscount = $format($discountedPrice);

    $symbolTag = "<span class='currency-symbol text-[0.75em] align-middle'>{$currencySymbol}</span>";

    if ($discountedPrice < $originalPrice) {
        if ($symbolOnRight) {
            return $cache[$cacheKey] = "<s class='old-price me-1 opacity-50 no-underline text-[0.6em] self-center'><span class='line-through'>{$formattedOriginal}</span>{$symbolTag}</s><span>{$formattedDiscount}{$symbolTag}</span>";
        }

        return $cache[$cacheKey] = "<s class='old-price me-1 opacity-50 no-underline text-[0.6em] self-center'>{$symbolTag}<span class='line-through'>{$formattedOriginal}</span></s><span>{$symbolTag}{$formattedDiscount}</span>";
    }

    return $cache[$cacheKey] = $symbolOnRight
        ? "{$formattedOriginal}{$symbolTag}"
        : "{$symbolTag}{$formattedOriginal}";
}

function newPlanDiscountPrice(?Plan $plan)
{
    static $cache = [];

    if (! $plan) {
        return null;
    }

    $planId = $plan->id;

    if (isset($cache[$planId])) {
        return $cache[$planId];
    }

    $price = $plan->price;

    if (\App\Helpers\Classes\MarketplaceHelper::isRegistered('discount-manager')) {
        $activeDiscounts = \App\Extensions\DiscountManager\System\Models\ConditionalDiscount::where('active', true)
            ->where('scheduled', false)
            ->whereNotNull('coupon_id')
            ->orderBy('amount', 'desc')
            ->get();

        $codes = Gateways::getCache(
            static fn () => Gateways::where('is_active', 1)
                ->pluck('code')
                ->implode(','),
            '-active-gateway-codes'
        );
        foreach ($activeDiscounts as $discount) {
            if (\App\Extensions\DiscountManager\System\Services\DiscountService::checkDiscountConditionsFor($discount, $planId, $codes)) {
                $discounted = $price - ($price * ($discount->amount / 100));

                return $cache[$planId] = $discounted;
            }
        }
    }

    return $cache[$planId] = $price;
}

function displayDiscountOrCouponName(?Plan $plan): ?string
{
    $title = checkCouponInRequest()?->name;

    if ($title) {
        return "<small class='text-gray-500 ms-1 text-3xs'>({$title})</small>";
    }

    return null;
}

function getVersion(?string $version = ''): string
{
    $fileVersion = file_get_contents(base_path('version.txt'));
    if ($fileVersion) {
        $version = trim($fileVersion);
    }
    $parts = explode('.', $version ?? '');
    if (count($parts) === 1) {
        $parts = [$parts[0], '0', '0'];
    } elseif (count($parts) === 2) {
        if (strlen($parts[1]) === 2) {
            $parts = [$parts[0], $parts[1][0], $parts[1][1]];
        } else {
            $parts[] = '0';
        }
    }
    $parts = array_map(static fn ($p) => (string) (int) $p, $parts);
    $parts = array_slice(array_pad($parts, 3, '0'), 0, 3);

    return implode('.', $parts);
}

function getMetaTitle($setting, $settingTwo, $ext_title = null)
{
    $ext_title = $ext_title == null ? ' | ' . __('Home') : $ext_title;
    $lang = app()->getLocale();

    if ($lang == $settingTwo->languages_default) {
        if (isset($setting->meta_title)) {
            $title = $setting->meta_title;
        } else {
            $title = $setting->site_name . $ext_title;
        }
    } else {
        $meta_title = PrivacyTerms::where('type', 'meta_title')->where('lang', $lang)->first();
        if ($meta_title) {
            $title = $meta_title->content;
        } else {

            if (isset($setting->meta_title)) {
                $title = $setting->meta_title;
            } else {
                $title = $setting->site_name . $ext_title;
            }
        }
    }

    return $title;
}

function getMetaDesc($setting, $settingTwo)
{
    $lang = app()->getLocale();

    if ($lang == $settingTwo->languages_default) {
        if (isset($setting->meta_description)) {
            $desc = $setting->meta_description;
        } else {
            $desc = '';
        }
    } else {
        $meta_description = PrivacyTerms::where('type', 'meta_desc')->where('lang', $lang)->first();
        if ($meta_description) {
            $desc = $meta_description->content;
        } else {

            if (isset($setting->meta_description)) {
                $desc = $setting->meta_description;
            } else {
                $desc = '';
            }
        }
    }

    return $desc;
}

function parseRSS($feed_url, $limit = 10): null|array|string
{

    try {
        $rss = simplexml_load_string(file_get_contents($feed_url));
        $posts = [];
        $id = 1;

        foreach ($rss->channel->item as $item) {
            if ($id > $limit) {
                break;
            }
            $posts[] = [
                'id'          => $id,
                'link'        => $item->link,
                'title'       => $item->title,
                'description' => $item->description,
                'image'       => $item->enclosure ? $item->enclosure['url'] : null,
            ];
            $id++;
        }

        return $posts;

    } catch (Exception $e) {
        return $e->getMessage();
    }

}

function isChatBot($id)
{

    if (! $id) {
        return false;
    }

    $isChatBot = App\Models\ChatBotHistory::where('user_openai_chat_id', $id)->first();

    if ($isChatBot) {
        return true;
    }

    return false;

}

function ThumbImage($source_file, $max_width = 800, $max_height = 800, $quality = 80, $convert_to_webp = true)
{
    if (setting('image_thumbnail') == 0 || Str::contains($source_file, 'http') || Str::contains($source_file, 'loading.svg')) {
        return $source_file;
    }

    try {
        $original_source_file = $source_file;
        $source_file = public_path($source_file);
        $disk = 'thumbs';
        $dst_dir = '';
        $thumbDisk = Storage::disk($disk);
        if (! $thumbDisk->exists($dst_dir)) {
            $thumbDisk->makeDirectory($dst_dir, $mode = 7777, true, true);
        }

        $is_url = filter_var($source_file, FILTER_VALIDATE_URL);
        $extension = strtolower(pathinfo($source_file, PATHINFO_EXTENSION));
        if ($is_url) {
            $url_parts = pathinfo($source_file);
            $extension = strtolower($url_parts['extension'] ?? '');
            $mime_map = [
                'jpg'  => 'image/jpeg',
                'jpeg' => 'image/jpeg',
                'png'  => 'image/png',
                'gif'  => 'image/gif',
                'webp' => 'image/webp',
            ];
            if (array_key_exists($extension, $mime_map)) {
                $mime = $mime_map[$extension];
            } else {
                $mime = mime_content_type($source_file);
            }
        } else {
            if (! file_exists($source_file)) {
                return $original_source_file;
            }

            if (is_dir($source_file)) {
                return $original_source_file;
            }

            $imgsize = getimagesize($source_file);
            if (! $imgsize) {
                return $original_source_file;
            }
            $width = $imgsize[0];
            $height = $imgsize[1];
            $mime = $imgsize['mime'];
        }
        $signature = 'v3|' . $source_file . '|' . ($is_url ? '' : @filemtime($source_file)) . '|' . $max_width . 'x' . $max_height . '|webp:' . (int) $convert_to_webp;
        $dst_file_name = pathinfo(basename($source_file), PATHINFO_FILENAME) . '_' . md5($signature);

        // Reuse an existing generated thumbnail for this exact signature.
        $candidateExtensions = array_values(array_filter(array_unique([
            $convert_to_webp ? 'webp' : null,
            $extension,
            'jpg',
            'jpeg',
            'png',
            'gif',
        ])));
        foreach ($candidateExtensions as $candidateExtension) {
            $candidatePath = $dst_dir . $dst_file_name . '.' . $candidateExtension;
            if ($thumbDisk->exists($candidatePath) && ($thumbDisk->size($candidatePath) ?? 0) > 0) {
                return $thumbDisk->url($candidatePath);
            }
        }

        if ($is_url) {
            $temp_source_file = tempnam(sys_get_temp_dir(), 'thumb');
            file_put_contents($temp_source_file, file_get_contents($source_file));
            $source_file = $temp_source_file;
        }

        switch ($mime) {
            case 'image/gif':
                $image_create = 'imagecreatefromgif';
                $image = 'imagegif';
                $extension = 'gif';
                $quality = null;

                break;

            case 'image/png':
                $image_create = 'imagecreatefrompng';
                $image = 'imagepng';
                $extension = 'png';
                $quality = max(0, min(9, (int) round((100 - (int) $quality) / 10)));

                break;

            case 'image/jpeg':
                $image_create = 'imagecreatefromjpeg';
                $image = 'imagejpeg';
                $extension = 'jpg';
                $quality = max(40, min(90, (int) $quality));

                break;
            case 'image/jpg':
                $image_create = 'imagecreatefromjpeg';
                $image = 'imagejpeg';
                $extension = 'jpg';
                $quality = max(40, min(90, (int) $quality));

                break;

            case 'image/webp':
                $image_create = 'imagecreatefromwebp';
                $image = 'imagewebp';
                $extension = 'webp';
                $quality = max(40, min(90, (int) $quality));

                break;

            default:
                if ($is_url) {
                    unlink($source_file);
                }

                return $original_source_file;

                break;
        }

        $webp_supported = function_exists('imagewebp') && function_exists('imagecreatefromwebp');
        $use_webp_output = $convert_to_webp && $webp_supported && $mime !== 'image/gif';

        $src_img = $image_create($source_file);
        if (! $src_img) {
            if ($is_url && file_exists($source_file)) {
                unlink($source_file);
            }

            return $original_source_file;
        }

        $width = imagesx($src_img);
        $height = imagesy($src_img);
        if ($width < 1 || $height < 1) {
            if ($is_url && file_exists($source_file)) {
                unlink($source_file);
            }
            imagedestroy($src_img);

            return $original_source_file;
        }

        $ratio = min($max_width / $width, $max_height / $height, 1);
        $target_width = max(1, (int) floor($width * $ratio));
        $target_height = max(1, (int) floor($height * $ratio));

        $dst_img = imagecreatetruecolor($target_width, $target_height);
        if (in_array($mime, ['image/png', 'image/gif', 'image/webp'])) {
            imagealphablending($dst_img, false);
            imagesavealpha($dst_img, true);
            $transparent = imagecolorallocatealpha($dst_img, 0, 0, 0, 127);
            imagefilledrectangle($dst_img, 0, 0, $target_width, $target_height, $transparent);
        }

        imagecopyresampled($dst_img, $src_img, 0, 0, 0, 0, $target_width, $target_height, $width, $height);

        $target_max = 300 * 1024;
        $output_extension = $use_webp_output ? 'webp' : $extension;
        $final_path = $thumbDisk->path($dst_dir . $dst_file_name . '.' . $output_extension);

        // Mild sharpening to avoid overly soft thumbnails after resize.
        if (function_exists('imageconvolution')) {
            @imageconvolution($dst_img, [
                [-1, -1, -1],
                [-1, 16, -1],
                [-1, -1, -1],
            ], 8, 0);
        }

        if ($use_webp_output) {
            $webp_quality_steps = [88, 84, 80, 76, 72];
            $saved = false;
            foreach ($webp_quality_steps as $webp_quality) {
                $write_result = imagewebp($dst_img, $final_path, $webp_quality);
                $filesize = @filesize($final_path) ?: 0;
                if ($write_result && $filesize > 0) {
                    $saved = true;
                }
                if ($saved && $filesize <= $target_max) {
                    break;
                }
            }
            if (! $saved) {
                $output_extension = $extension;
                $final_path = $thumbDisk->path($dst_dir . $dst_file_name . '.' . $output_extension);
                if ($mime === 'image/png') {
                    $saved = (bool) imagepng($dst_img, $final_path, max(0, min(9, (int) $quality)));
                } elseif ($mime === 'image/gif') {
                    $saved = (bool) imagegif($dst_img, $final_path);
                } else {
                    $saved = (bool) imagejpeg($dst_img, $final_path, 84);
                }
                if (! $saved || ! file_exists($final_path) || (@filesize($final_path) ?: 0) < 1) {
                    if ($is_url && file_exists($source_file)) {
                        unlink($source_file);
                    }
                    imagedestroy($dst_img);
                    imagedestroy($src_img);

                    return $original_source_file;
                }
            }
        } elseif ($mime === 'image/gif') {
            $saved = (bool) $image($dst_img, $final_path);
            if (! $saved || ! file_exists($final_path) || (@filesize($final_path) ?: 0) < 1) {
                if ($is_url && file_exists($source_file)) {
                    unlink($source_file);
                }
                imagedestroy($dst_img);
                imagedestroy($src_img);

                return $original_source_file;
            }
        } elseif ($mime === 'image/png') {
            $compressionLevels = array_values(array_unique([
                (int) $quality,
                4,
                6,
                8,
                9,
            ]));

            $saved = false;
            foreach ($compressionLevels as $compressionLevel) {
                $write_result = $image($dst_img, $final_path, max(0, min(9, $compressionLevel)));
                $filesize = @filesize($final_path) ?: 0;
                if ($write_result && $filesize > 0) {
                    $saved = true;
                }
                if ($saved && $filesize <= $target_max) {
                    break;
                }
            }
            if (! $saved) {
                if ($is_url && file_exists($source_file)) {
                    unlink($source_file);
                }
                imagedestroy($dst_img);
                imagedestroy($src_img);

                return $original_source_file;
            }
        } else {
            $qualitySteps = array_values(array_unique([
                (int) $quality,
                88,
                84,
                80,
                76,
                72,
                68,
            ]));
            rsort($qualitySteps);

            $saved = false;
            foreach ($qualitySteps as $qualityStep) {
                $write_result = $image($dst_img, $final_path, max(68, min(92, $qualityStep)));
                $filesize = @filesize($final_path) ?: 0;
                if ($write_result && $filesize > 0) {
                    $saved = true;
                }
                if ($saved && $filesize <= $target_max) {
                    break;
                }
            }
            if (! $saved) {
                if ($is_url && file_exists($source_file)) {
                    unlink($source_file);
                }
                imagedestroy($dst_img);
                imagedestroy($src_img);

                return $original_source_file;
            }
        }

        if ($is_url && file_exists($source_file)) {
            unlink($source_file);
        }

        if ($dst_img) {
            imagedestroy($dst_img);
        }
        if ($src_img) {
            imagedestroy($src_img);
        }

        return $thumbDisk->url($dst_file_name . '.' . $output_extension);
    } catch (Exception $e) {
        return $original_source_file ?? $source_file;
    }

}

function PurgeThumbImages(): void
{
    $dst_dir = public_path('uploads/thumbnail/');

    try {
        \Illuminate\Support\Facades\File::deleteDirectory($dst_dir);
    } catch (\Exception $e) {
        Log::error($e->getMessage());
    }
}

function convertHistoryToGemini($history): array
{
    $convertedHistory = [];
    $system = false;
    foreach ($history as $item) {
        $role = $item['role'];
        switch ($role) {
            case 'assistant':
                $role = 'model';

                break;
            case 'system':
                $system = true;
                $role = 'user';

                break;
            default:
                $role = 'user';

                break;
        }

        $content = $item['content'];

        if (is_string($content)) {
            $convertedItem = [
                'parts' => [
                    [
                        'text' => $content,
                    ],
                ],
                'role' => $role,
            ];
        } elseif (is_array($content)) {
            $isImage = (bool) array_filter($content, function ($item) {
                return isset($item['type']) && $item['type'] === 'image_url';
            });

            if ($isImage) {
                $convertedItem = [
                    'parts' => [],
                ];

                foreach ($content as $contentItem) {

                    if ($contentItem['type'] === 'text') {
                        $convertedItem['parts'][] = [
                            'text' => $item['text'],
                        ];
                    } elseif ($contentItem['type'] === 'image_url') {
                        $convertedItem['parts'][] = [
                            'inline_data' => [
                                'mime_type' => extractMimeType($contentItem['image_url']['url']),
                                'data'      => extractBase64ImageData($contentItem['image_url']['url']),
                            ],
                        ];
                    }

                }

            }
        }

        $convertedHistory[] = $convertedItem;

        if ($system) {
            // add system back message
            $convertedItem = [
                'parts' => [
                    [
                        'text' => 'Yes, sure.',
                    ],
                ],
                'role' => 'model',
            ];
            $convertedHistory[] = $convertedItem;
            $system = false;
        }
    }

    return adjustHistory($convertedHistory);
}

function extractMimeType($data)
{
    $pattern = '/^data:(image\/[a-z]+);base64,/';
    if (preg_match($pattern, $data, $matches)) {
        return $matches[1]; // MIME türünü döndürür
    }

    return null; // Eşleşme bulunamazsa null döndürür
}

function extractBase64ImageData($dataURI)
{
    // JPEG veya PNG için MIME türlerini destekleyen regex deseni
    $pattern = '/^data:image\/(?:jpeg|png);base64,/';
    if (preg_match($pattern, $dataURI, $matches)) {
        // Regex ile başlık bulunursa, başlığı atıp sadece Base64 verisini döndür
        return substr($dataURI, strlen($matches[0]));
    }

    return null; // Uygun başlık bulunamazsa null döndür
}

function adjustHistory($history)
{
    $adjustedHistory = [];

    // Iterate through the history
    foreach ($history as $index => $item) {
        // Add the current item to the adjusted history
        $adjustedHistory[] = $item;

        if (isset($history[$index + 1]['role']) and isset($item['role'])) {
            // Check if the next item exists and has the same role as the current item
            if (isset($history[$index + 1]) && $history[$index + 1]['role'] === $item['role']) {
                // If the next item has the same role, add the opposite role message after it
                $adjustedHistory[] = [
                    'role'  => ($item['role'] === 'user' ? 'model' : 'user'),
                    'parts' => [
                        [
                            'text' => 'This is a placeholder message.',
                        ],
                    ],
                ];
            }
        }

    }

    return $adjustedHistory;
}

if (! function_exists('viewName')) {
    function viewName(string $viewName): string
    {
        return $viewName;
    }
}

function showTeamFunctionality(): bool
{
    $shouldCheck = Helper::appIsDemo();

    if ($shouldCheck) {
        return true;
    }

    $plan = Plan::getCache(
        static fn () => Plan::query()->where('is_team_plan', 1)->first(),
        '_team'
    );

    return Helper::setting('team_functionality') && ! auth()?->user()?->getAttribute('team_id') && $plan;
}

if (! function_exists('mimeToExtension')) {
    function mimeToExtension(string $mimeType): string
    {
        static $mimeMap = [
            // Microsoft Office
            'application/msword'                                                        => 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'   => 'docx',
            'application/vnd.ms-excel'                                                  => 'xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'         => 'xlsx',
            'application/vnd.ms-powerpoint'                                             => 'ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation' => 'pptx',

            // Documents
            'application/pdf'  => 'pdf',
            'text/plain'       => 'txt',
            'text/csv'         => 'csv',
            'text/html'        => 'html',
            'text/css'         => 'css',
            'text/javascript'  => 'js',
            'application/json' => 'json',
            'application/xml'  => 'xml',
            'text/xml'         => 'xml',
            'application/rtf'  => 'rtf',

            // Images
            'image/jpeg'    => 'jpg',
            'image/png'     => 'png',
            'image/gif'     => 'gif',
            'image/webp'    => 'webp',
            'image/svg+xml' => 'svg',
            'image/bmp'     => 'bmp',
            'image/tiff'    => 'tiff',
            'image/ico'     => 'ico',

            // Audio
            'audio/mpeg' => 'mp3',
            'audio/wav'  => 'wav',
            'audio/ogg'  => 'ogg',
            'audio/mp4'  => 'm4a',
            'audio/aac'  => 'aac',

            // Video
            'video/mp4'       => 'mp4',
            'video/mpeg'      => 'mpeg',
            'video/quicktime' => 'mov',
            'video/x-msvideo' => 'avi',
            'video/webm'      => 'webm',

            // Archives
            'application/zip'              => 'zip',
            'application/x-rar-compressed' => 'rar',
            'application/x-tar'            => 'tar',
            'application/gzip'             => 'gz',
            'application/x-7z-compressed'  => '7z',

            // Programming
            'text/x-php'         => 'php',
            'text/x-python'      => 'py',
            'text/x-java-source' => 'java',
            'text/x-c'           => 'c',
            'text/x-c++src'      => 'cpp',
            'text/x-csharp'      => 'cs',

            // Other
            'application/octet-stream' => 'bin',
        ];

        // First check our mapping
        if (isset($mimeMap[$mimeType])) {
            return $mimeMap[$mimeType];
        }

        // Fallback: extract extension from MIME type
        $parts = explode('/', $mimeType);
        if (count($parts) >= 2) {
            $subtype = $parts[1];
            // Handle special cases in subtype
            $subtype = str_replace([
                'vnd.ms-',
                'vnd.openxmlformats-officedocument.',
                'x-',
            ], '', $subtype);

            return $subtype;
        }

        return 'bin';
    }
}

function generateRandomWords($wordCount): string
{
    $words = [];
    for ($i = 0; $i < $wordCount; $i++) {
        $words[] = \Illuminate\Support\Str::random(random_int(2, 5));
    }

    return implode(' ', $words);
}

if (! function_exists('isDBDriverSQLite')) {
    function isDBDriverSQLite(): bool
    {
        return DB::connection() instanceof \Illuminate\Database\SQLiteConnection;
    }
}

if (! function_exists('class_interface_exists')) {
    /**
     * @param  string<class-string>  $class
     * @param  string<class-string>  $interface
     */
    function class_interface_exists(string $class, string $interface): bool
    {
        return in_array($interface, class_implements($class), true);
    }
}

if (! function_exists('getClientIp')) {
    function getClientIp(): string
    {
        return request()?->server('HTTP_CF_CONNECTING_IP') ?? request()?->ip();
    }
}

if (! function_exists('number_shorten')) {
    function number_shorten($number, $precision = 3)
    {
        $suffixes = ['', __('K'), __('M'), __('B'), __('T'), __('Qa'), __('Qi')];
        $index = (int) log(abs($number), 1000);
        $index = max(0, min(count($suffixes) - 1, $index));
        $shortNumber = $number / (1000 ** $index);
        $formatted = rtrim(rtrim(number_format($shortNumber, $precision), '0'), '.');

        return $formatted . $suffixes[$index];
    }
}

if (! function_exists('getSocialMediaIcon')) {
    function getSocialMediaIcon(?string $socialMediaName)
    {
        $image = '';

        switch ($socialMediaName) {
            case 'instagram':
                $image =
                    '<svg width="20" height="20" viewBox="0 0 20 20" fill="url(#paint0_linear_2140_988)" xmlns="http://www.w3.org/2000/svg"> <path d="M10.0215 5.03711C12.7285 5.03711 14.9629 7.27148 14.9629 9.97852C14.9629 12.7285 12.7285 14.9199 10.0215 14.9199C7.27148 14.9199 5.08008 12.7285 5.08008 9.97852C5.08008 7.27148 7.27148 5.03711 10.0215 5.03711ZM10.0215 13.2012C11.7832 13.2012 13.2012 11.7832 13.2012 9.97852C13.2012 8.2168 11.7832 6.79883 10.0215 6.79883C8.2168 6.79883 6.79883 8.2168 6.79883 9.97852C6.79883 11.7832 8.25977 13.2012 10.0215 13.2012ZM16.2949 4.86523C16.2949 5.50977 15.7793 6.02539 15.1348 6.02539C14.4902 6.02539 13.9746 5.50977 13.9746 4.86523C13.9746 4.2207 14.4902 3.70508 15.1348 3.70508C15.7793 3.70508 16.2949 4.2207 16.2949 4.86523ZM19.5605 6.02539C19.6465 7.61523 19.6465 12.3848 19.5605 13.9746C19.4746 15.5215 19.1309 16.8535 18.0137 18.0137C16.8965 19.1309 15.5215 19.4746 13.9746 19.5605C12.3848 19.6465 7.61523 19.6465 6.02539 19.5605C4.47852 19.4746 3.14648 19.1309 1.98633 18.0137C0.869141 16.8535 0.525391 15.5215 0.439453 13.9746C0.353516 12.3848 0.353516 7.61523 0.439453 6.02539C0.525391 4.47852 0.869141 3.10352 1.98633 1.98633C3.14648 0.869141 4.47852 0.525391 6.02539 0.439453C7.61523 0.353516 12.3848 0.353516 13.9746 0.439453C15.5215 0.525391 16.8965 0.869141 18.0137 1.98633C19.1309 3.10352 19.4746 4.47852 19.5605 6.02539ZM17.498 15.6504C18.0137 14.4043 17.8848 11.3965 17.8848 9.97852C17.8848 8.60352 18.0137 5.5957 17.498 4.30664C17.1543 3.49023 16.5098 2.80273 15.6934 2.50195C14.4043 1.98633 11.3965 2.11523 10.0215 2.11523C8.60352 2.11523 5.5957 1.98633 4.34961 2.50195C3.49023 2.8457 2.8457 3.49023 2.50195 4.30664C1.98633 5.5957 2.11523 8.60352 2.11523 9.97852C2.11523 11.3965 1.98633 14.4043 2.50195 15.6504C2.8457 16.5098 3.49023 17.1543 4.34961 17.498C5.5957 18.0137 8.60352 17.8848 10.0215 17.8848C11.3965 17.8848 14.4043 18.0137 15.6934 17.498C16.5098 17.1543 17.1973 16.5098 17.498 15.6504Z"/> <defs> <linearGradient id="paint0_linear_2140_988" x1="0.375" y1="10" x2="19.625" y2="10" gradientUnits="userSpaceOnUse"> <stop stop-color="#EB6434"/> <stop offset="0.545" stop-color="#BB2D9F"/> <stop offset="0.98" stop-color="#BB802D"/> </linearGradient> </defs> </svg>';

                break;
            case 'x':
                $image =
                    '<svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor" xmlns="http://www.w3.org/2000/svg"> <path d="M10.7124 7.62177L17.4133 0H15.8254L10.0071 6.61788L5.35992 0H0L7.02738 10.0074L0 18H1.58799L7.73237 11.0113L12.6401 18H18L10.7121 7.62177H10.7124ZM8.53747 10.0956L7.82546 9.09906L2.16017 1.16971H4.59922L9.17118 7.56895L9.8832 8.56546L15.8262 16.8835H13.3871L8.53747 10.096V10.0956Z"/> </svg>';

                break;
            case 'tiktok':
                $image =
                    '<svg width="18" height="18" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="96" height="96" rx="21" fill="black" style="fill:black;fill-opacity:1;"/><path d="M73.31 25.7456C72.785 25.4743 72.274 25.1769 71.7788 24.8545C70.3389 23.9025 69.0186 22.7808 67.8465 21.5135C64.9139 18.158 63.8186 14.7538 63.4151 12.3705H63.4313C63.0943 10.3921 63.2337 9.11214 63.2547 9.11214H49.8974V60.7624C49.8974 61.4558 49.8974 62.1412 49.8682 62.8185C49.8682 62.9027 49.8601 62.9805 49.8553 63.0712C49.8553 63.1085 49.8553 63.1474 49.8472 63.1863C49.8472 63.196 49.8472 63.2057 49.8472 63.2154C49.7064 65.0686 49.1123 66.8588 48.1173 68.4286C47.1222 69.9983 45.7566 71.2994 44.1407 72.2175C42.4565 73.1757 40.5517 73.6782 38.614 73.6757C32.3906 73.6757 27.3468 68.6011 27.3468 62.334C27.3468 56.0669 32.3906 50.9923 38.614 50.9923C39.7921 50.9912 40.9629 51.1766 42.083 51.5415L42.0992 37.9412C38.6989 37.502 35.2444 37.7722 31.9538 38.7348C28.6631 39.6975 25.6077 41.3317 22.9802 43.5343C20.678 45.5346 18.7425 47.9214 17.2608 50.5872C16.6969 51.5594 14.5695 55.4658 14.3119 61.8058C14.1499 65.4044 15.2306 69.1326 15.7458 70.6734V70.7058C16.0699 71.6132 17.3256 74.7094 19.372 77.3197C21.0221 79.4135 22.9716 81.2527 25.1579 82.7783V82.7459L25.1903 82.7783C31.6567 87.1724 38.8263 86.884 38.8263 86.884C40.0674 86.8338 44.2249 86.884 48.9463 84.6464C54.183 82.1658 57.1642 78.47 57.1642 78.47C59.0688 76.2618 60.5832 73.7452 61.6426 71.0282C62.8513 67.8509 63.2547 64.0401 63.2547 62.5171V35.1155C63.4168 35.2127 65.5749 36.6401 65.5749 36.6401C65.5749 36.6401 68.6842 38.633 73.5352 39.9309C77.0155 40.8544 81.7045 41.0488 81.7045 41.0488V27.7887C80.0615 27.9669 76.7255 27.4485 73.31 25.7456Z" fill="white" style="fill:white;fill-opacity:1;"/></svg>';

                break;
            case 'linkedin':
                $image =
                    '<svg width="16" height="16" viewBox="0 0 16 16" fill="#0077B5" xmlns="http://www.w3.org/2000/svg"><path d="M3.88281 15.4209H0.794922V5.49316H3.88281V15.4209ZM2.32227 4.16504C1.35938 4.16504 0.5625 3.33496 0.5625 2.33887C0.5625 1.37598 1.35938 0.579102 2.32227 0.579102C3.31836 0.579102 4.11523 1.37598 4.11523 2.33887C4.11523 3.33496 3.31836 4.16504 2.32227 4.16504ZM12.3496 15.4209V10.6064C12.3496 9.44434 12.3164 7.9834 10.7227 7.9834C9.12891 7.9834 8.89648 9.21191 8.89648 10.5068V15.4209H5.80859V5.49316H8.76367V6.85449H8.79688C9.22852 6.09082 10.2246 5.26074 11.7188 5.26074C14.8398 5.26074 15.4375 7.31934 15.4375 9.97559V15.4209H12.3496Z"/></svg>';

                break;
            case 'facebook':
                $image =
                    '<svg width="10" height="18" viewBox="0 0 10 18" fill="#1877F2" xmlns="http://www.w3.org/2000/svg"><path d="M8.96777 10.0625H6.47754V17.5H3.15723V10.0625H0.43457V7.00781H3.15723V4.65039C3.15723 1.99414 4.75098 0.5 7.1748 0.5C8.33691 0.5 9.56543 0.732422 9.56543 0.732422V3.35547H8.2041C6.87598 3.35547 6.47754 4.15234 6.47754 5.01562V7.00781H9.43262L8.96777 10.0625Z"/></svg>';

                break;
            default:
                $image =
                    '<svg width="20" height="20" viewBox="0 0 20 20" fill="url(#paint0_linear_2140_988)" xmlns="http://www.w3.org/2000/svg"> <path d="M10.0215 5.03711C12.7285 5.03711 14.9629 7.27148 14.9629 9.97852C14.9629 12.7285 12.7285 14.9199 10.0215 14.9199C7.27148 14.9199 5.08008 12.7285 5.08008 9.97852C5.08008 7.27148 7.27148 5.03711 10.0215 5.03711ZM10.0215 13.2012C11.7832 13.2012 13.2012 11.7832 13.2012 9.97852C13.2012 8.2168 11.7832 6.79883 10.0215 6.79883C8.2168 6.79883 6.79883 8.2168 6.79883 9.97852C6.79883 11.7832 8.25977 13.2012 10.0215 13.2012ZM16.2949 4.86523C16.2949 5.50977 15.7793 6.02539 15.1348 6.02539C14.4902 6.02539 13.9746 5.50977 13.9746 4.86523C13.9746 4.2207 14.4902 3.70508 15.1348 3.70508C15.7793 3.70508 16.2949 4.2207 16.2949 4.86523ZM19.5605 6.02539C19.6465 7.61523 19.6465 12.3848 19.5605 13.9746C19.4746 15.5215 19.1309 16.8535 18.0137 18.0137C16.8965 19.1309 15.5215 19.4746 13.9746 19.5605C12.3848 19.6465 7.61523 19.6465 6.02539 19.5605C4.47852 19.4746 3.14648 19.1309 1.98633 18.0137C0.869141 16.8535 0.525391 15.5215 0.439453 13.9746C0.353516 12.3848 0.353516 7.61523 0.439453 6.02539C0.525391 4.47852 0.869141 3.10352 1.98633 1.98633C3.14648 0.869141 4.47852 0.525391 6.02539 0.439453C7.61523 0.353516 12.3848 0.353516 13.9746 0.439453C15.5215 0.525391 16.8965 0.869141 18.0137 1.98633C19.1309 3.10352 19.4746 4.47852 19.5605 6.02539ZM17.498 15.6504C18.0137 14.4043 17.8848 11.3965 17.8848 9.97852C17.8848 8.60352 18.0137 5.5957 17.498 4.30664C17.1543 3.49023 16.5098 2.80273 15.6934 2.50195C14.4043 1.98633 11.3965 2.11523 10.0215 2.11523C8.60352 2.11523 5.5957 1.98633 4.34961 2.50195C3.49023 2.8457 2.8457 3.49023 2.50195 4.30664C1.98633 5.5957 2.11523 8.60352 2.11523 9.97852C2.11523 11.3965 1.98633 14.4043 2.50195 15.6504C2.8457 16.5098 3.49023 17.1543 4.34961 17.498C5.5957 18.0137 8.60352 17.8848 10.0215 17.8848C11.3965 17.8848 14.4043 18.0137 15.6934 17.498C16.5098 17.1543 17.1973 16.5098 17.498 15.6504Z"/> <defs> <linearGradient id="paint0_linear_2140_988" x1="0.375" y1="10" x2="19.625" y2="10" gradientUnits="userSpaceOnUse"> <stop stop-color="#EB6434"/> <stop offset="0.545" stop-color="#BB2D9F"/> <stop offset="0.98" stop-color="#BB802D"/> </linearGradient> </defs> </svg>';

                break;
        }

        return $image;
    }
}

if (! function_exists('getFuncArgs')) {
    /**
     * get filled args, we can use this function when we have a lot of optional parameters on function
     *
     * @param  callable|string  $function
     */
    function getFuncArgs(object $obj, $function, array $args): array
    {
        $reflectionFunc = new ReflectionMethod($obj, $function);
        $parameters = $reflectionFunc->getParameters();

        $filledParams = [];

        foreach ($parameters as $index => $param) {
            if (isset($args[$index]) && ($args[$index] != null && $args[$index] != '')) {
                $paramName = $param->getName();
                $filledParams[$paramName] = $args[$index];
            }
        }

        return $filledParams;
    }
}
if (! function_exists('deepMerge')) {
    function deepMerge(array $array1, array $array2): array
    {
        foreach ($array2 as $key => $value) {
            if (is_array($value) && isset($array1[$key]) && is_array($array1[$key])) {
                $array1[$key] = deepMerge($array1[$key], $value);
            } else {
                $array1[$key] = $value;
            }
        }

        return $array1;
    }
}

if (! function_exists('lightenColor')) {
    function lightenColor($hex, $percent = 30)
    {
        $hex = ltrim($hex, '#');
        $r = hexdec(substr($hex, 0, 2));
        $g = hexdec(substr($hex, 2, 2));
        $b = hexdec(substr($hex, 4, 2));
        $r = (int) ($r + ((255 - $r) * $percent) / 100);
        $g = (int) ($g + ((255 - $g) * $percent) / 100);
        $b = (int) ($b + ((255 - $b) * $percent) / 100);

        return sprintf('#%02x%02x%02x', $r, $g, $b);
    }
}

if (! function_exists('isFileSecure')) {
    function isFileSecure($file): bool
    {
        $extension = strtolower($file->guessExtension());
        $clientExtension = strtolower(pathinfo($file->getClientOriginalName(), PATHINFO_EXTENSION));

        $dangerousExtensions = [
            'php', 'php3', 'php4', 'php5', 'phtml', 'phps',
            'asp', 'aspx', 'jsp', 'jspx',
            'exe', 'bat', 'cmd', 'com', 'pif', 'scr',
            'vbs', 'vb', 'js', 'jse', 'ws', 'wsf', 'wsc', 'wsh',
            'ps1', 'ps1xml', 'ps2', 'ps2xml', 'psc1', 'psc2',
            'msh', 'msh1', 'msh2', 'mshxml', 'msh1xml', 'msh2xml',
            'scf', 'lnk', 'inf', 'reg', 'jar',
        ];

        if (in_array($extension, $dangerousExtensions) ||
            in_array($clientExtension, $dangerousExtensions)) {
            return false;
        }

        $mimeType = $file->getMimeType();
        $dangerousMimeTypes = [
            // Executables & binaries
            'application/x-msdownload', // .exe
            'application/x-msdos-program',
            'application/x-ms-installer', // .msi
            'application/x-executable',
            'application/x-sharedlib',
            'application/x-dosexec',

            // Scripts
            'application/javascript',
            'application/x-javascript',
            'text/javascript',
            'application/x-php',
            'text/x-php',
            'application/x-python',
            'text/x-python',
            'application/x-ruby',
            'text/x-ruby',
            'application/x-perl',
            'text/x-perl',
            'application/x-sh',
            'application/x-csh',

            // Batch & shell
            'text/x-shellscript',
            'application/x-bat',
            'application/x-msbatch',

            // Other potentially dangerous
            'application/x-object',
            'application/x-mach-binary',
        ];

        if (in_array($mimeType, $dangerousMimeTypes, true)) {
            return false;
        }

        $filename = $file->getClientOriginalName();
        if (preg_match('/[<>:"|?*\x00-\x1f]/', $filename)) {
            return false;
        }

        return true;
    }

    if (! function_exists('processFileContent')) {
        function processFileContent(string $extension, string $tempFileName): string
        {
            $tempPath = public_path("uploads/$tempFileName");

            switch ($extension) {
                case 'pdf':
                    $parser = new \Smalot\PdfParser\Parser;
                    $text = $parser->parseFile($tempPath)->getText();

                    return mb_check_encoding($text, 'UTF-8') ? $text : mb_convert_encoding($text, 'UTF-8', mb_detect_encoding($text));

                case 'docx':
                    return docxToText($tempPath);

                case 'doc':
                    return docToText($tempPath);

                case 'csv':
                    $file = file_get_contents($tempPath);
                    $rows = explode(PHP_EOL, $file);
                    $header = str_getcsv(array_shift($rows));
                    $dataAsJson = [];
                    foreach ($rows as $row) {
                        if (trim($row)) {
                            $data = array_combine($header, array_pad(str_getcsv($row), count($header), ''));
                            $dataAsJson[] = json_encode($data, JSON_THROW_ON_ERROR);
                        }
                    }

                    return implode("\n", $dataAsJson);

                case 'xls':
                case 'xlsx':
                    $parser = app(ParserExcelService::class);

                    return $parser->setPath($tempPath)->parse();

                default:
                    throw new RuntimeException('Unsupported file type');
            }
        }
    }

    if (! function_exists('docToText')) {
        function docToText($path_to_file): array|string|null
        {
            $fileHandle = fopen($path_to_file, 'rb');
            $line = @fread($fileHandle, filesize($path_to_file));
            $lines = explode(chr(0x0D), $line);
            $response = '';
            foreach ($lines as $current_line) {
                $pos = strpos($current_line, chr(0x00));
                if (($pos !== false) || ($current_line === '')) {
                    $response .= "\n";
                } else {
                    $response .= $current_line . ' ';
                }
            }

            return preg_replace('/[^a-zA-Z0-9\s\,\.\-\n\r\t@\/\_\(\)]/', '', $response);
        }
    }

    if (! function_exists('docxToText')) {
        function docxToText($path_to_file): bool|string
        {
            $response = '';
            $zip = new ZipArchive;
            if (! $zip->open($path_to_file)) {
                return false;
            }
            for ($i = 0; $i < $zip->numFiles; $i++) {
                $entry = $zip->statIndex($i);
                if ($entry['name'] !== 'word/document.xml') {
                    continue;
                }
                $content = $zip->getFromIndex($i);

                $content = str_replace(['</w:r></w:p></w:tc><w:tc>', '</w:r></w:p>'], ["\r\n", "\n"], $content ?? '');
                $content = strip_tags($content);
                $response .= $content;
            }
            $zip->close();

            return $response;
        }
    }

    if (! function_exists('validateUploadedFile')) {
        function validateUploadedFile(string $filePath, string $extension): bool
        {
            if (! file_exists($filePath)) {
                return false;
            }

            $content = file_get_contents($filePath, false, null, 0, 1024);

            $dangerousPatterns = [
                '/<\?php/',
                '/<\?=/',
                '/<script/',
                '/eval\s*\(/',
                '/exec\s*\(/',
                '/system\s*\(/',
                '/shell_exec/',
                '/passthru/',
            ];

            foreach ($dangerousPatterns as $pattern) {
                if (preg_match($pattern, $content)) {
                    return false;
                }
            }

            if (in_array($extension, ['jpg', 'jpeg', 'png', 'gif', 'webp'])) {
                $imageInfo = @getimagesize($filePath);
                if ($imageInfo === false) {
                    return false;
                }
            }

            return true;
        }
    }
}

if (! function_exists('processSecureFileUpload')) {
    function processSecureFileUpload($file, string $basePath): ?string
    {
        if (! $file) {
            return null;
        }

        $fileName = $file->getClientOriginalName();

        // Security check first
        if (! isFileSecure($file)) {
            throw new RuntimeException(__("File '{$fileName}' is not allowed for security reasons."));
        }

        // Calculate hash of new file
        $newHash = md5_file($file->getRealPath());

        // Get all files in the user's folder using Storage
        $existingFiles = Storage::disk('public')->files($basePath);

        foreach ($existingFiles as $existingFilePath) {
            $fullPath = Storage::disk('public')->path($existingFilePath);

            if (file_exists($fullPath)) {
                $existingHash = md5_file($fullPath);

                // Found file with same content - reuse it
                if ($existingHash === $newHash) {
                    return '/uploads/' . $existingFilePath;
                }
            }
        }

        return '/uploads/' . $file->store($basePath, 'public');
    }
}

if (! function_exists('measureVoiceLength')) {
    function measureVoiceLength($filePath): ?float
    {
        try {
            $getID3 = new getID3;
            $fileInfo = $getID3->analyze($filePath);

            if (isset($fileInfo['playtime_seconds'])) {
                return round($fileInfo['playtime_seconds'] / 60, 2);
            }

            return 1;
        } catch (Throwable $e) {
            return 1;
        }
    }
}

if (! function_exists('autoTranslateLanguage')) {
    /**
     * Auto-translate English language file to target language
     *
     * @param  string  $targetLang  Target language code (e.g., 'fr', 'es', 'de')
     * @param  int  $batchSize  Number of keys to translate per batch (default: 100)
     *
     * @return array Returns status information about the translation process
     */
    function autoTranslateLanguage(string $targetLang, int $batchSize = 100): array
    {
        try {
            // Define file paths
            $enJsonFilePath = base_path('lang/en.json');
            $targetJsonFilePath = base_path("lang/{$targetLang}.json");

            // Check if English file exists
            if (! file_exists($enJsonFilePath)) {
                return [
                    'success'          => false,
                    'message'          => 'English language file (lang/en.json) not found.',
                    'translated_count' => 0,
                ];
            }

            // Load English translations
            $enTranslations = json_decode(file_get_contents($enJsonFilePath), true);

            if (json_last_error() !== JSON_ERROR_NONE) {
                return [
                    'success'          => false,
                    'message'          => 'Invalid JSON format in English language file.',
                    'translated_count' => 0,
                ];
            }

            // Load or create target language file
            $targetTranslations = [];
            if (file_exists($targetJsonFilePath)) {
                $targetTranslations = json_decode(file_get_contents($targetJsonFilePath), true);
                if (json_last_error() !== JSON_ERROR_NONE) {
                    $targetTranslations = [];
                }
            }

            // Initialize missing keys with null values
            $missingKeys = array_diff_key($enTranslations, $targetTranslations);
            foreach ($missingKeys as $key => $value) {
                $targetTranslations[$key] = null;
            }

            // Find untranslated keys (null or empty values)
            $untranslatedKeys = array_filter($targetTranslations, function ($value, $key) {
                return $value === null || $value === '';
            }, ARRAY_FILTER_USE_BOTH);

            // Limit to batch size
            $keysToTranslate = array_slice(array_keys($untranslatedKeys), 0, $batchSize);

            if (empty($keysToTranslate)) {
                return [
                    'success'          => true,
                    'message'          => 'All translations are already completed.',
                    'translated_count' => 0,
                ];
            }

            $translatedCount = 0;
            $errors = [];

            // Translate each key
            foreach ($keysToTranslate as $key) {
                try {
                    // Get the English text to translate
                    $englishText = $enTranslations[$key] ?? $key;

                    // Skip if empty
                    if (empty(trim($englishText))) {
                        $englishText = $key;
                    }

                    // Translate using Google Translate
                    $result = GoogleTranslate::withSource('en')
                        ->withTarget($targetLang)
                        ->translate($englishText);

                    $translatedText = $result->getTranslatedText();

                    // Update target translations
                    $targetTranslations[$key] = $translatedText;
                    $translatedCount++;

                } catch (Throwable $e) {
                    $errors[] = "Failed to translate key '{$key}': " . $e->getMessage();

                    continue;
                }
            }

            // Save updated translations to file
            $jsonContent = json_encode($targetTranslations, JSON_PRETTY_PRINT|JSON_UNESCAPED_UNICODE);

            if (file_put_contents($targetJsonFilePath, $jsonContent) === false) {
                return [
                    'success'          => false,
                    'message'          => 'Failed to write to target language file.',
                    'translated_count' => $translatedCount,
                    'errors'           => $errors,
                ];
            }

            return [
                'success' => true,
                'message' => $translatedCount > 0 ?
                    "Successfully translated {$translatedCount} keys to {$targetLang}." :
                    'No new translations were needed.',
                'translated_count' => $translatedCount,
                'errors'           => $errors,
                'remaining_keys'   => count($untranslatedKeys) - $translatedCount,
            ];

        } catch (Throwable $e) {
            return [
                'success'          => false,
                'message'          => 'Translation process failed: ' . $e->getMessage(),
                'translated_count' => 0,
            ];
        }
    }
}

if (! function_exists('getSelectedOrUploadedFile')) {
    /**
     * Get the selected file path or upload a new one.
     *
     * @param  bool  $full  Return full URL or relative path
     */
    function getSelectedOrUploadedFile(UploadedFile $uploadedFile, bool $full = true): ?string
    {
        if (! isFileSecure($uploadedFile)) {
            return null;
        }

        $userId = auth()?->id();

        $fileDirectory = match ($uploadedFile->getClientMimeType()) {
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp' => 'images',
            'audio/mpeg', 'audio/wav', 'audio/ogg'                            => 'voices',
            default                                                           => 'others',
        };

        $baseDirectory = $userId
            ? "/uploads/media/{$fileDirectory}/u-{$userId}/"
            : '/uploads/guest/';

        $absoluteBasePath = public_path($baseDirectory);

        if (! File::exists($absoluteBasePath)) {
            File::makeDirectory($absoluteBasePath, 0755, true);
        }

        $originalName = pathinfo($uploadedFile->getClientOriginalName(), PATHINFO_FILENAME);
        $extension = strtolower($uploadedFile->guessExtension() ?: 'bin');
        $safeBaseName = preg_replace('/[^a-zA-Z0-9_\-]/', '_', $originalName);
        $fileName = "{$safeBaseName}.{$extension}";
        $absolutePath = $absoluteBasePath . $fileName;

        if ($extension === 'bin') {
            return null;
        }

        if (File::exists($absolutePath)) {
            $existingHash = sha1_file($absolutePath);
            $currentHash = sha1_file($uploadedFile->getRealPath());

            if ($existingHash === $currentHash) {
                $relative = $baseDirectory . $fileName;

                return $full ? url($relative) : $relative;
            }

            $timestamp = now()->format('YmdHis');
            $fileName = "{$safeBaseName}_{$timestamp}.{$extension}";
            $absolutePath = $absoluteBasePath . $fileName;
        }

        $uploadedFile->move($absoluteBasePath, $fileName);

        if (! validateUploadedFile($absolutePath, $extension)) {
            File::delete($absolutePath);

            return null;
        }

        $relative = $baseDirectory . $fileName;

        return $full ? url($relative) : $relative;
    }
}

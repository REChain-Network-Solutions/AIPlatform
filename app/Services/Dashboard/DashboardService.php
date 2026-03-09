<?php

namespace App\Services\Dashboard;

use App\Enums\Plan\FrequencyEnum;
use App\Models\Activity;
use App\Models\DashboardWidget;
use App\Models\Finance\Subscription;
use App\Models\Referer;
use App\Models\Setting;
use App\Models\Usage;
use App\Models\User;
use App\Models\UserActivity;
use App\Models\UserAffiliate;
use App\Models\UserOpenai;
use App\Models\UserOrder;
use App\Models\UserSupport;
use Carbon\Carbon;
use Exception;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\support\Str;

class DashboardService
{
    public int $cacheTtl = 300;

    public array $userActiveOrderStatuses = ['Success', 'Approved'];

    public array $frequencies = [
        'lifetime'          => 0,
        'yearly'            => 0,
        'monthly'           => 0,
        'prepaid'           => 0,
    ];

    public array $randomColors = ['#74DB84', '#74A9DB', '#DB9374', '#8185F44D', '#E3E8E8', '#C674DB'];

    public function latestOrders(): Collection|array
    {
        return UserOrder::query()
            ->with('user', 'plan')
            ->orderBy('created_at', 'desc')
            ->take(10)
            ->get();
    }

    public function activity(): Collection|array
    {
        return Activity::query()
            ->with('user:id,name,surname,avatar')
            ->orderBy('created_at', 'desc')
            ->get();
    }

    public function cache($key, $value)
    {
        return Cache::remember($key, $this->cacheTtl, $value);
    }

    public function setCache(): void
    {
        $this->setDashboardWidgets()
            ->setWhatIsNew()
            ->setFinance()
            ->setDailySales()
            ->setPopularPlansData()
            ->setAPICostDistribution()
            ->setTopCountries()
            ->setUsage()
            ->setCostManagement()
            ->setNewCustomers()
            ->setUserAndPlatform()
            ->setMostUsedLastOpenAITools()
            ->setUserTraffic()
            ->setGeneratedContent()
            ->setUserBehaviorData()
            ->setRecentActivity()
            ->setAvailableDiskspace()
            ->setUsersDetail()
            ->setDailyActivity();
    }

    private function isMobileDevice($userAgent): bool
    {
        $pattern = '/Mobile|Android|Silk\/|Kindle|BlackBerry|Opera Mini|Opera Mobi/i';

        return (bool) preg_match($pattern, $userAgent);
    }

    public function setDailyUsers(): static
    {
        $this->cache('daily_users', function () {
            return User::query()
                ->select(DB::raw('count(*) as total'), DB::raw("DATE_FORMAT(created_at,'%Y-%m-%d') as days"))
                ->groupBy('days')
                ->get()
                ->toJson();
        });

        return $this;
    }

    public function setDailyUsages(): static
    {
        $this->cache('daily_usages', function () {
            return UserOpenai::query()
                ->select(
                    DB::raw('SUM(IF(credits=1,credits,0)) as sumsImage'),
                    DB::raw('SUM(IF(credits>1,credits,0)) as sumsWord'),
                    DB::raw("DATE_FORMAT(created_at,'%Y-%m-%d') as days")
                )->groupBy('days')
                ->get();
        });

        return $this;
    }

    // top countries
    public function setTopCountries(): static
    {
        $this->cache('top_countries', function () {
            return User::query()
                ->select('country', DB::raw('count(*) as total'))
                ->groupBy('country')
                ->orderByDesc('total')
                ->get()
                ->toJson();
        });

        return $this;
    }

    //  daily sales = daily earing
    public function setDailySales(): static
    {
        $this->cache('daily_sales', function () {
            return UserOrder::query()
                ->select(DB::raw('sum(price) as sums'), DB::raw("DATE_FORMAT(created_at,'%Y-%m-%d') as days"))
                ->groupBy('days')
                ->get()
                ->toJson();
        });

        return $this;
    }

    // each plans label and colors
    public function planLabelsAndColors(): array
    {
        return [
            FrequencyEnum::MONTHLY->value => [
                'label' => FrequencyEnum::MONTHLY->label(),
                'color' => FrequencyEnum::MONTHLY->color(),
            ],
            FrequencyEnum::YEARLY->value => [
                'label' => FrequencyEnum::YEARLY->label(),
                'color' => FrequencyEnum::YEARLY->color(),
            ],
            FrequencyEnum::LIFETIME_MONTHLY->value => [
                'label' => FrequencyEnum::LIFETIME_MONTHLY->label(),
                'color' => FrequencyEnum::LIFETIME_MONTHLY->color(),
            ],
            FrequencyEnum::LIFETIME_YEARLY->value => [
                'label' => FrequencyEnum::LIFETIME_YEARLY->label(),
                'color' => FrequencyEnum::LIFETIME_YEARLY->color(),
            ],
            FrequencyEnum::LIFETIME->value => [
                'label' => FrequencyEnum::LIFETIME->label(),
                'color' => FrequencyEnum::LIFETIME->color(),
            ],
            FrequencyEnum::PREPAID->value => [
                'label' => FrequencyEnum::PREPAID->label(),
                'color' => FrequencyEnum::PREPAID->color(),
            ],
        ];
    }

    // daily activity
    public function setDailyActivity(): static
    {
        $this->cache('daily_activity_this_week', function () {
            return UserActivity::where('created_at', '>=', Carbon::today()->subDays(7))->count();
        });

        $this->cache('daily_activity_last_week', function () {
            $startOfLastWeek = Carbon::now()->subWeek()->startOfWeek();
            $endOfLastWeek = Carbon::now()->subWeek()->endOfWeek();

            return UserActivity::whereBetween('created_at', [$startOfLastWeek, $endOfLastWeek])->count();
        });

        $this->cache('total_daily_activity', function () {
            return UserActivity::all()->count();
        });

        return $this;
    }

    // finance
    public function setFinance(): static
    {
        $this->cache('referral_payout', function () {
            return UserAffiliate::whereNot('status', 'Waiting')->sum('amount');
        });

        return $this;
    }

    // what is new
    public function setWhatIsNew(): static
    {
        $this->cache('tickets', function () {
            return UserSupport::count();
        });

        $this->cache('transactions', function () {
            return UserOrder::count();
        });

        $this->cache('documents', function () {
            return UserOpenai::count();
        });

        return $this;
    }

    // api cost distribution
    public function setAPICostDistribution(): static
    {
        $this->cache('api_cost_distribution', function () {
            $userTotalOpenAICount = UserOpenai::count();

            return UserOpenai::query()
                ->with('generator')
                ->select('engine', DB::raw('COUNT(*) as total'))
                ->groupBy('engine')
                ->orderBy('total', 'desc')
                ->limit(5)
                ->get()
                ->map(function (UserOpenai $item, $key) use ($userTotalOpenAICount) {
                    $title = $item->engine ? $item->engine->label() : 'Unknown';
                    $percentage = $userTotalOpenAICount ? round(($item->getAttribute('total') / $userTotalOpenAICount) * 100) : 0;

                    return [
                        'label' => $title,
                        'value' => $percentage,
                    ];
                });
        });

        return $this;
    }

    // cost management
    public function setCostManagement(): static
    {
        $totalEarn = Cache::get('total_sales') ?? 0;
        $totalUser = Cache::get('total_users') ?? 0;
        $totalSpend = setting('total_spend', 0);

        // Convert possible formatted numbers ("1,006.63") to float
        $totalEarn = (float) str_replace(',', '', $totalEarn);
        $totalSpend = (float) str_replace(',', '', $totalSpend);
        $totalUser = (float) str_replace(',', '', $totalUser);

        // Cache cost per user
        $this->cache('cost_per_user', function () use ($totalSpend, $totalUser) {
            return $totalUser == 0 ? 0 : round($totalSpend / $totalUser, 1);
        });

        $this->cache('income_per_user', function () use ($totalEarn, $totalUser) {
            return $totalUser == 0 ? 0 : round($totalEarn / $totalUser, 1);
        });

        $this->cache('net_profit', function () {
            return Cache::get('income_per_user') - Cache::get('cost_per_user');
        });

        return $this;
    }

    // new customer
    public function setNewCustomers(): static
    {
        $this->cache('new_customers', function () {
            $tenDaysAgo = Carbon::today()->subDays(9);

            $users = User::query()
                ->select('*', DB::raw("DATE_FORMAT(created_at, '%Y-%m-%d') as day"))
                ->where('created_at', '>=', $tenDaysAgo)->get();
            $dailyUsers = [];

            for ($day = $tenDaysAgo; $day <= Carbon::today(); $day->addDay()) {
                $formattedDay = $day->format('Y-m-d');
                $dailyUsers[$formattedDay] = [
                    'date' => $formattedDay,
                    'paid' => 0,
                    'free' => 0,
                ];
            }

            foreach ($users as $user) {
                if ($user->activePlan()) {
                    $dailyUsers[$user->day]['paid'] += 1;
                } else {
                    $dailyUsers[$user->day]['free'] += 1;
                }
            }

            return json_encode($dailyUsers);
        });

        return $this;
    }

    // recent transactions
    public function getRecentTransactions()
    {
        $this->cache('recent_transactions_enabled', function () {
            $setting = Setting::getCache();

            return (bool) $setting?->bank_transfer_active;
        });

        if (Cache::get('recent_transactions_enabled', false)) {
            return $this->cache('recent_transactions', function () {
                return UserOrder::where('payment_type', 'banktransfer')->orderByRaw("CASE WHEN status = 'Waiting' THEN 0 ELSE 1 END")->orderBy('created_at', 'desc')->get();
            });
        }

        return [];
    }

    // user and platform
    public function setUserAndPlatform(): static
    {
        $lastWeek = Carbon::today()->subWeek();
        $startOfLastWeek = Carbon::today()->subWeek()->startOfWeek();
        $endOfLastWeek = Carbon::today()->subWeek()->endOfWeek();

        $this->cache('this_week_new_users', function () use ($lastWeek) {
            return User::where('created_at', '>=', $lastWeek)
                ->groupBy('email')
                ->count();
        });

        $this->cache('last_week_new_users', function () use ($startOfLastWeek, $endOfLastWeek) {
            return User::whereBetween('created_at', [$startOfLastWeek, $endOfLastWeek])
                ->groupBy('email')
                ->count();
        });

        $this->cache('this_week_subscribers', function () use ($lastWeek) {
            return Subscription::where('created_at', '>=', $lastWeek)
                ->groupBy('user_id')
                ->count();
        });

        $this->cache('last_week_subscribers', function () use ($startOfLastWeek, $endOfLastWeek) {
            return Subscription::whereBetween('created_at', [$startOfLastWeek, $endOfLastWeek])
                ->groupBy('user_id')
                ->count();
        });

        $this->cache('this_week_referrals', function () use ($lastWeek) {
            return User::where('created_at', '>=', $lastWeek)
                ->whereNotNull('affiliate_code')
                ->count();
        });

        $this->cache('last_week_referrals', function () use ($startOfLastWeek, $endOfLastWeek) {
            return User::whereBetween('created_at', [$startOfLastWeek, $endOfLastWeek])
                ->whereNotNull('affiliate_code')
                ->count();
        });

        $this->cache('this_week_total_users', function () use ($lastWeek) {
            return UserActivity::where('created_at', '>=', $lastWeek)
                ->groupBy('email')
                ->count();
        });

        $this->cache('last_week_total_users', function () use ($startOfLastWeek, $endOfLastWeek) {
            return UserActivity::whereBetween('created_at', [$startOfLastWeek, $endOfLastWeek])
                ->groupBy('email')
                ->count();
        });

        return $this;
    }

    // generated content
    public function setGeneratedContent(): static
    {
        $this->cache('generated_content', function () {
            return UserOpenai::all();
        });

        return $this;
    }

    // user traffic
    public function setUserTraffic(): static
    {
        $this->cache('user_traffic', function () {
            return Referer::select('domain', DB::raw('COUNT(*) as users'))
                ->groupBy('domain')
                ->orderByDesc('users')
                ->take(5)
                ->get()
                ->toJson();
        });

        return $this;
    }

    // recent activity
    public function setRecentActivity(): static
    {
        $this->cache('recent_activity', function () {
            return UserOrder::query()
                ->with('user', 'plan')
                ->orderBy('created_at', 'desc')
                ->take(5)
                ->get();
        });

        return $this;
    }

    // available disk
    public function setAvailableDiskspace(): static
    {
        try {
            $this->cache('available_diskspace', function () {
                $total_space = disk_total_space('/');
                $free_space = disk_free_space('/');

                return round(($free_space / $total_space) * 100);
            });
        } catch (Exception $e) {
            $this->cache('available_diskspace', function () {
                return 0;
            });
        }

        return $this;
    }

    /**
     * @todo implement online user logic
     */
    public function setUsersDetail(): static
    {
        $this->cache('total_user', function () {
            return User::count();
        });

        $this->cache('paid_user', function () {
            $users = User::all();
            $paidUser = $users->filter(fn ($entity) => (bool) $entity->activePlan())->count();

            return $paidUser;
        });

        $this->cache('free_user', function () {
            return Cache::get('total_user', 0) - Cache::get('paid_user', 0);
        });

        $this->cache('trial_user', function () {
            return Subscription::whereNotNull('trial_ends_at')->count();
        });

        Cache::remember('online_user', 120, function () {
            return User::query()->where('last_activity_at', '>=', now()->subMinutes(2))->count();
        });

        return $this;
    }

    // last used ai tools
    public function setMostUsedLastOpenAITools(): static
    {
        $this->cache('popular_tools_data', function () {
            $userTotalOpenAICount = UserOpenai::query()->count('id');

            return UserOpenai::query()
                ->with('generator')
                ->select('openai_id', DB::raw('COUNT(*) as total'))
                ->groupBy('openai_id')
                ->orderBy('total', 'desc')
                ->limit(5)
                ->get()
                ->map(function (UserOpenai $item, $key) use ($userTotalOpenAICount) {
                    $color = $this->randomColors[$key] ?? '#000000';

                    $openai = $item->getAttribute('generator');

                    $title = $openai ? $openai->getAttribute('title') : 'Unknown';

                    $percentage = $userTotalOpenAICount ? round(($item->getAttribute('total') / $userTotalOpenAICount) * 100) : 0;

                    return [
                        'label' => $title,
                        'value' => $percentage,
                        'color' => $color,
                    ];
                });
        });

        return $this;
    }

    // popular plans
    public function setPopularPlansData(): static
    {
        $this->cache('popular_plans_data', function () {
            $approvedOrders = UserOrder::query()
                ->with('plan')
                ->whereIn('status', $this->userActiveOrderStatuses)
                ->get();

            // Başlangıçta tüm frekansları 0 olarak başlatıyoruz.
            $plan_counts = array_fill_keys(array_keys($this->frequencies), 0);

            foreach ($approvedOrders as $order) {
                $plan = $order->plan;

                if ($plan) {
                    $key = ($plan->type === 'subscription') ? $plan->frequency : $plan->type;

                    if (isset($plan_counts[$key])) {
                        if (Str::startsWith($key, 'lifetime')) {
                            $plan_counts['lifetime']++;
                        } else {
                            $plan_counts[$key]++;
                        }
                    }
                }
            }

            $plan_names_colors = $this->planLabelsAndColors();
            $popular_plans_data = [];

            foreach ($plan_counts as $key => $count) {
                $popular_plans_data[] = [
                    'label' => $plan_names_colors[$key]['label'],
                    'value' => $count,
                    'color' => $count > 0 ? $plan_names_colors[$key]['color'] : '#2C36490D',
                ];
            }

            return $popular_plans_data;
        });

        return $this;
    }

    // usages
    public function setUsage(): static
    {
        $usage = $this->cache('instance_usage', function () {
            return Usage::getSingle();
        });

        foreach ([
            'sales_this_week'      => $usage->this_week_sales,
            'sales_previous_week'  => $usage->last_week_sales,
            'words_this_week'      => $usage->this_week_word_count,
            'words_previous_week'  => $usage->last_week_word_count,
            'images_this_week'     => $usage->this_week_image_count,
            'images_previous_week' => $usage->last_week_image_count,
            'users_this_week'      => $usage->this_week_user_count,
            'users_previous_week'  => $usage->last_week_user_count,
            'usage_this_week'      => $usage->this_week_word_count + $usage->this_week_image_count,
            'usage_previous_week'  => $usage->last_week_word_count + $usage->last_week_image_count,
            'total_sales'          => $usage->total_sales,
            'total_usage'          => $usage->total_word_count + $usage->total_image_count,
            'total_users'          => $usage->total_user_count,
        ] as $key => $value) {
            Cache::put($key, $value, $this->cacheTtl);
        }

        return $this;
    }

    // user behavors on mobile and desktop
    public function setUserBehaviorData(): static
    {
        $this->cache('user_behavior_data', function () {
            $activities = UserActivity::query()->select('connection')->get();

            $mobileCount = $activities->filter(fn ($activity) => $this->isMobileDevice($activity->connection))->count();

            $desktopCount = $activities->count() - $mobileCount;

            return [
                [
                    'label' => 'Mobile',
                    'value' => $mobileCount,
                    'color' => 'hsl(var(--primary))',
                ],
                [
                    'label' => 'Desktop',
                    'value' => $desktopCount,
                    'color' => 'hsl(var(--secondary))',
                ],
            ];
        });

        return $this;
    }

    // set widgets
    public function setDashboardWidgets(): static
    {
        $this->cache('dashboard_widgets', function () {
            return DashboardWidget::query()->orderBy('order')->get();
        });

        return $this;
    }
}

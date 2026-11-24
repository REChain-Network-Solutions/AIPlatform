# Faz 6: Gelismis Ozellikler

> **Durum:** Bekliyor
> **Bagimlilik:** Faz 4
> **Referans:** [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) Bolum 3.2, 3.11
> **Not:** Her alt adim bagimsiz olarak eklenebilir

## 6.1: Kredi Suresi Dolma (Expiration)

`shared_credits_expires_at` kolonu Faz 1'de users tablosuna eklendi.

### Scheduled Command

```php
// app/Console/Commands/ExpireSharedCredits.php
class ExpireSharedCredits extends Command
{
    protected $signature = 'shared-credits:expire';
    protected $description = 'Expire shared credits past their expiration date';

    public function handle(SharedCreditService $service): void
    {
        $users = User::query()
            ->where('credit_system_type', 'shared')
            ->where('shared_credits', '>', 0)
            ->whereNotNull('shared_credits_expires_at')
            ->where('shared_credits_expires_at', '<=', now())
            ->get();

        foreach ($users as $user) {
            $expiredAmount = $user->shared_credits;
            $user->shared_credits = 0;
            $user->shared_credits_expires_at = null;
            $user->save();

            $service->logTransaction(
                user: $user,
                entity: null,
                actionType: 'expire',
                amount: -$expiredAmount,
                balanceAfter: 0,
                description: 'Credits expired'
            );
        }

        $this->info("Expired credits for {$users->count()} users.");
    }
}
```

**Schedule:** `app/Console/Kernel.php`
```php
$schedule->command('shared-credits:expire')->daily();
```

**Uyari bildirimi (opsiyonel):** Expiration'dan X gun once uyari:
```php
// app/Console/Commands/SharedCreditExpiryWarning.php
$schedule->command('shared-credits:expiry-warning')->daily();
// 3 gun kala: "Krediniz 3 gun icinde dolacak"
```

### Plan Subscribe Entegrasyonu

```php
// CreditUpdater::creditIncreaseSubscribePlan() icinde:
if ($expirationDays = (int) setting('shared_credit_expiration_days')) {
    $user->shared_credits_expires_at = now()->addDays($expirationDays);
    $user->save();
}
```

### Settings (Faz 3'te eklenir)
```php
'shared_credit_expiration_days' => 0, // 0 = son kullanma yok, 30 = 30 gun sonra dolacak
```

## 6.2: Otomatik Top-Up

Belirli esik altina dusunce otomatik odeme + kredi ekleme.

### Veritabani

```php
// users tablosuna yeni kolonlar (migration)
$table->boolean('shared_credit_auto_topup')->default(false);
$table->decimal('shared_credit_topup_threshold', 12, 4)->default(100);
$table->decimal('shared_credit_topup_amount', 12, 4)->default(1000);
$table->string('shared_credit_topup_payment_method')->nullable(); // Stripe PM id
```

### SharedCreditService Entegrasyonu

```php
// SharedCreditService::deduct() sonunda basarili dusme sonrasi:
if ($freshUser->shared_credit_auto_topup
    && $freshUser->shared_credits < $freshUser->shared_credit_topup_threshold) {
    dispatch(new AutoTopUpJob($freshUser));
}
```

### AutoTopUpJob

```php
// app/Jobs/AutoTopUpJob.php
class AutoTopUpJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public function __construct(public User $user) {}

    public function handle(SharedCreditService $service): void
    {
        // Tekrar kontrol (race condition)
        $this->user->refresh();
        if ($this->user->shared_credits >= $this->user->shared_credit_topup_threshold) {
            return; // Bakiye doldurulmus
        }

        $amount = $this->user->shared_credit_topup_amount;
        $paymentMethod = $this->user->shared_credit_topup_payment_method;

        if (!$paymentMethod) {
            $this->user->notify(new AutoTopUpFailedNotification('No payment method'));
            return;
        }

        try {
            // Mevcut Stripe/PayPal altyapisi ile odeme
            // PaymentIntent::create() veya benzeri
            // Basarili ise:
            $service->topUp($this->user, $amount, 'Auto top-up', null, 'auto_topup');
        } catch (\Exception $e) {
            $this->user->notify(new AutoTopUpFailedNotification($e->getMessage()));
        }
    }
}
```

### Kullanici Ayarlari UI

```blade
{{-- Dashboard Settings veya Credit sayfasi --}}
<x-card x-data="{ enabled: {{ auth()->user()->shared_credit_auto_topup ? 'true' : 'false' }} }">
    <h4>{{ __('Auto Top-up') }}</h4>

    <x-forms.input type="checkbox" name="shared_credit_auto_topup"
        :checked="auth()->user()->shared_credit_auto_topup"
        :label="__('Enable automatic top-up')"
        x-model="enabled"
        switcher
    />

    <div x-show="enabled" x-transition class="mt-3 space-y-3">
        <x-forms.input type="number" name="shared_credit_topup_threshold"
            :value="auth()->user()->shared_credit_topup_threshold ?? 100"
            :label="__('When balance falls below')"
        />

        <x-forms.input type="number" name="shared_credit_topup_amount"
            :value="auth()->user()->shared_credit_topup_amount ?? 1000"
            :label="__('Top-up amount')"
        />
    </div>
</x-card>
```

## 6.3: Kullanim Limitleri (Feature Limits)

`shared_credit_feature_limits` JSON alani Faz 1'de plans tablosuna eklendi.

### JSON Yapisi

```json
{
    "daily_image_limit": 50,
    "daily_video_limit": 5,
    "monthly_image_limit": 500,
    "monthly_video_limit": 30,
    "per_model_limits": {
        "gpt-5-pro": { "daily": 100, "monthly": 1000 },
        "sora-2-pro": { "daily": 2, "monthly": 10 }
    }
}
```

### HasCreditLimit Entegrasyonu

```php
// HasCreditLimit::decreaseCredit() icinde (shared guard'da), dusme oncesi:
if ($this->isSharedCreditUser()) {
    $user = $this->getUser();
    $featureLimits = $user->activePlan()?->shared_credit_feature_limits;

    if ($featureLimits && $this->isFeatureLimitExceeded($user, $this->creditEnum(), $featureLimits)) {
        return false; // Limit asildi
    }

    // ... normal deduct logigi ...
}
```

```php
private function isFeatureLimitExceeded(User $user, EntityEnum $entity, array $limits): bool
{
    $tokenType = $entity->tokenType()->value; // AITokenType: 'image', 'text_to_video', vb.

    // Gunluk limit kontrolu
    $dailyKey = "daily_{$tokenType}_limit";
    if (isset($limits[$dailyKey])) {
        $todayCount = SharedCreditTransaction::where('user_id', $user->id)
            ->where('feature_type', $tokenType)
            ->where('action_type', 'deduct')
            ->whereDate('created_at', today())
            ->count();
        if ($todayCount >= $limits[$dailyKey]) {
            return true;
        }
    }

    // Aylik limit kontrolu
    $monthlyKey = "monthly_{$tokenType}_limit";
    if (isset($limits[$monthlyKey])) {
        $monthCount = SharedCreditTransaction::where('user_id', $user->id)
            ->where('feature_type', $tokenType)
            ->where('action_type', 'deduct')
            ->where('created_at', '>=', now()->startOfMonth())
            ->count();
        if ($monthCount >= $limits[$monthlyKey]) {
            return true;
        }
    }

    // Model bazli limit
    $entitySlug = $entity->slug();
    if (isset($limits['per_model_limits'][$entitySlug])) {
        $modelLimits = $limits['per_model_limits'][$entitySlug];

        if (isset($modelLimits['daily'])) {
            $todayModelCount = SharedCreditTransaction::where('user_id', $user->id)
                ->where('entity_key', $entitySlug)
                ->where('action_type', 'deduct')
                ->whereDate('created_at', today())
                ->count();
            if ($todayModelCount >= $modelLimits['daily']) {
                return true;
            }
        }

        // monthly benzer sekilde...
    }

    return false;
}
```

### Admin Plan Form'unda Limit Ayarlari (Faz 2/3 ile baglantili)

Plan olusturma/duzenleme formunda `shared_credit_feature_limits` JSON editor'u:

```blade
<div x-show="$wire.credit_system_type === 'shared'" x-transition>
    <h5>{{ __('Feature Limits (optional)') }}</h5>
    <div class="grid grid-cols-2 gap-4">
        <x-forms.input type="number" name="daily_image_limit" :label="__('Daily image limit')" />
        <x-forms.input type="number" name="daily_video_limit" :label="__('Daily video limit')" />
        <x-forms.input type="number" name="monthly_image_limit" :label="__('Monthly image limit')" />
        <x-forms.input type="number" name="monthly_video_limit" :label="__('Monthly video limit')" />
    </div>
</div>
```

## 6.4: Bildirimler

### Notification Siniflari

```php
// app/Notifications/SharedCredit/
LowBalanceNotification.php           // Bakiye esik altina dustugunde
ExpiringCreditNotification.php       // Kredi suresi dolmadan X gun once
AutoTopUpFailedNotification.php      // Otomatik top-up basarisiz
WeeklyUsageReportNotification.php    // Haftalik kullanim raporu
```

Mail, database, ve push channel destegi.

### Dusuk Bakiye Alarmi

```php
// SharedCreditService::deduct() sonunda:
$thresholdPercent = (int) setting('shared_credit_low_balance_threshold', 20);
$planCredits = $user->activePlan()?->shared_credits_amount ?? 0;

if ($planCredits > 0) {
    $percent = ($freshUser->shared_credits / $planCredits) * 100;
    if ($percent <= $thresholdPercent && $percent + ($amount / $planCredits * 100) > $thresholdPercent) {
        // Esik yeni gecildi (tekrar tekrar bildirim gonderilmesin)
        $user->notify(new LowBalanceNotification($freshUser->shared_credits, $planCredits));
    }
}
```

### Haftalik Kullanim Raporu

```php
// app/Console/Commands/SendWeeklyUsageReport.php
$schedule->command('shared-credits:weekly-report')->weeklyOn(1, '9:00');

// Rapor icerigi:
// - Bu hafta harcanan toplam kredi
// - En cok kullanilan modeller (top 5)
// - Kalan bakiye
// - Feature bazli breakdown (words, images, video, tts)
```

## 6.5: API Endpoint'leri

Sanctum auth ile korunmus REST API.

### Route'lar

```php
// routes/api.php
Route::middleware('auth:sanctum')->prefix('v1/shared-credit')->group(function () {
    Route::get('/balance', [SharedCreditApiController::class, 'balance']);
    Route::get('/history', [SharedCreditApiController::class, 'history']);
    Route::get('/cost/{entity}', [SharedCreditApiController::class, 'cost']);
});
```

### API Resource Siniflari

```php
// app/Http/Resources/SharedCredit/
BalanceResource.php          // balance, credit_system_type, expires_at, plan
TransactionResource.php      // id, entity_key, action_type, amount, balance_after, created_at
CostResource.php             // entity_key, label, feature_type, shared_credit_index, has_db_override
```

### Response Ornekleri

**GET /api/v1/shared-credit/balance**
```json
{
    "data": {
        "balance": 8450.5,
        "credit_system_type": "shared",
        "expires_at": "2026-05-01T00:00:00Z",
        "plan": {
            "name": "Pro Plan",
            "shared_credits_amount": 10000
        }
    }
}
```

**GET /api/v1/shared-credit/history?per_page=10&action_type=deduct**
```json
{
    "data": [
        {
            "id": 123,
            "entity_key": "gpt_4_o",
            "action_type": "deduct",
            "amount": -100.0,
            "balance_after": 8450.5,
            "feature_type": "word",
            "created_at": "2026-04-01T12:00:00Z"
        }
    ],
    "meta": { "current_page": 1, "total": 150, "per_page": 10 }
}
```

**GET /api/v1/shared-credit/cost/dall_e_3**
```json
{
    "data": {
        "entity_key": "dall_e_3",
        "label": "DALL-E 3",
        "feature_type": "image",
        "shared_credit_index": 40.0,
        "has_db_override": false,
        "tooltip": "40 credits per image"
    }
}
```

## 6.6: Top-up Paketleri (Kredi Satin Alma)

v2 planindaki `shared_credit_packages` tablosu.

### Veritabani

```php
Schema::create('shared_credit_packages', function (Blueprint $table) {
    $table->id();
    $table->string('name');                  // 'Starter Pack', 'Pro Pack'
    $table->decimal('credits', 12, 4);      // Verilen kredi miktari
    $table->decimal('price', 10, 2);        // Fiyat
    $table->string('currency')->default('USD');
    $table->decimal('bonus_credits', 12, 4)->default(0);
    $table->boolean('is_active')->default(true);
    $table->boolean('is_featured')->default(false);
    $table->integer('sort_order')->default(0);
    $table->text('description')->nullable();
    $table->string('stripe_product_id')->nullable();
    $table->string('stripe_price_id')->nullable();
    $table->timestamps();
});
```

### Model

```php
// app/Models/SharedCreditPackage.php
class SharedCreditPackage extends Model
{
    protected $fillable = [
        'name', 'credits', 'price', 'currency', 'bonus_credits',
        'is_active', 'is_featured', 'sort_order', 'description',
        'stripe_product_id', 'stripe_price_id',
    ];

    protected $casts = [
        'credits' => 'float',
        'price' => 'float',
        'bonus_credits' => 'float',
        'is_active' => 'boolean',
        'is_featured' => 'boolean',
    ];

    public function scopeActive($query) { return $query->where('is_active', true); }
    public function totalCredits(): float { return $this->credits + $this->bonus_credits; }
}
```

### Admin CRUD

- **Controller:** `app/Http/Controllers/Admin/Finance/SharedCreditPackageController.php`
- Standard CRUD: index, create, store, edit, update, destroy
- **Route'lar:** `dashboard.admin.finance.shared-credit-packages.*`

### Kullanici Satin Alma Sayfasi

```blade
{{-- resources/views/default/panel/user/shared-credit/packages.blade.php --}}
<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    @foreach($packages as $package)
    <x-card :class="$package->is_featured ? 'border-primary ring-1 ring-primary' : ''">
        @if($package->is_featured)
            <x-badge class="bg-primary text-primary-foreground mb-2">{{ __('Popular') }}</x-badge>
        @endif
        <h3>{{ $package->name }}</h3>
        <div class="text-3xl font-bold text-primary mt-2">
            {{ number_format($package->totalCredits()) }}
        </div>
        <span class="text-foreground/60">{{ __('credits') }}</span>
        @if($package->bonus_credits > 0)
            <div class="text-green-500 text-sm mt-1">
                +{{ number_format($package->bonus_credits) }} {{ __('bonus') }}
            </div>
        @endif
        <div class="text-xl mt-3">
            {{ $package->currency }} {{ number_format($package->price, 2) }}
        </div>
        <x-button tag="a" href="{{ route('dashboard.user.shared-credit.purchase', $package) }}" class="mt-3 w-full">
            {{ __('Buy Now') }}
        </x-button>
    </x-card>
    @endforeach
</div>
```

Mevcut payment gateway altyapisi (Stripe, PayPal, vb.) kullanilarak satin alma yapilir.

## Ilgili Dosyalar

| Dosya | Islem | Alt Ozellik |
|-------|-------|-------------|
| `app/Console/Commands/ExpireSharedCredits.php` | CREATE | 6.1 |
| `app/Console/Commands/SharedCreditExpiryWarning.php` | CREATE | 6.1 |
| `app/Jobs/AutoTopUpJob.php` | CREATE | 6.2 |
| `app/Notifications/SharedCredit/LowBalanceNotification.php` | CREATE | 6.4 |
| `app/Notifications/SharedCredit/ExpiringCreditNotification.php` | CREATE | 6.4 |
| `app/Notifications/SharedCredit/AutoTopUpFailedNotification.php` | CREATE | 6.4 |
| `app/Notifications/SharedCredit/WeeklyUsageReportNotification.php` | CREATE | 6.4 |
| `app/Http/Controllers/Api/SharedCreditApiController.php` | CREATE | 6.5 |
| `app/Http/Resources/SharedCredit/BalanceResource.php` | CREATE | 6.5 |
| `app/Http/Resources/SharedCredit/TransactionResource.php` | CREATE | 6.5 |
| `app/Http/Resources/SharedCredit/CostResource.php` | CREATE | 6.5 |
| `app/Http/Controllers/Admin/Finance/SharedCreditPackageController.php` | CREATE | 6.6 |
| `app/Models/SharedCreditPackage.php` | CREATE | 6.6 |
| `database/migrations/` | CREATE | packages tablo, auto_topup kolonlari |
| `routes/api.php` | MODIFY | 6.5 |
| `routes/panel.php` | MODIFY | 6.6 |
| `app/Console/Kernel.php` | MODIFY | 6.1 schedule |
| `app/Services/SharedCredit/SharedCreditService.php` | MODIFY | 6.2, 6.3, 6.4 hook'lari |
| `app/Domains/Entity/Concerns/HasCreditLimit.php` | MODIFY | 6.3 feature limit kontrolu |

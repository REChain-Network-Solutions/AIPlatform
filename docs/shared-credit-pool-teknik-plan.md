# Shared Credit Pool (Paylasilmis Kredi Havuzu) - Teknik Implementasyon Plani

> **Tarih:** 2026-03-30
> **Durum:** Taslak - Onay Bekleniyor
> **Referans:** [Higgsfield Pricing](https://higgsfield.ai/pricing)

---

## Icindekiler

1. [Mevcut Sistem Analizi](#1-mevcut-sistem-analizi)
2. [Shared Credit Pool Mimarisi](#2-shared-credit-pool-mimarisi)
3. [Veritabani Degisiklikleri](#3-veritabani-degisiklikleri)
4. [Backend Implementasyonu](#4-backend-implementasyonu)
5. [Admin Panel Degisiklikleri](#5-admin-panel-degisiklikleri)
6. [Frontend / UX Degisiklikleri](#6-frontend--ux-degisiklikleri)
7. [Migrasyon Stratejisi](#7-migrasyon-stratejisi)
8. [Guvenlik ve Performans](#8-guvenlik-ve-performans)
9. [Faz Plani ve Oncelikler](#9-faz-plani-ve-oncelikler)
10. [Riskler ve Cozum Onerileri](#10-riskler-ve-cozum-onerileri)

---

## 1. Mevcut Sistem Analizi

### 1.1 Mevcut Kredi Yapisi (Separated Credits)

Simdi her AI modeli icin **ayri ayri kredi** tanimlaniyor. Kullanicinin `entity_credits` JSON alani su yapida:

```json
{
  "openai": {
    "gpt-4o": { "credit": 1000, "isUnlimited": false },
    "gpt-4o-mini": { "credit": 5000, "isUnlimited": false }
  },
  "stable_diffusion": {
    "dall-e-3": { "credit": 200, "isUnlimited": false }
  },
  "anthropic": {
    "claude-opus-4-6": { "credit": 500, "isUnlimited": true }
  }
}
```

**Sorun:** Kullanici GPT-4o kredisi bitince, DALL-E kredisi olsa bile kullanamaz. Her model icin ayri bakiye yonetimi var.

### 1.2 Kredi Dusme Akisi

```
Kullanici istek yapar
  -> Entity::driver($model)->forUser($user)
  -> input($data)->calculateCredit()
  -> hasCreditBalanceForInput() kontrolu
  -> AI API cagrisi yapilir
  -> decreaseCredit() ile kredi dusurulur
  -> UserUsageCredit kaydina log yazilir
```

### 1.3 Kritik Dosyalar

| Dosya | Rol |
|-------|-----|
| `app/Domains/Entity/Concerns/HasCreditLimit.php` | Kredi kontrolu ve dusme mantigi |
| `app/Domains/Entity/Concerns/Calculate/Has*.php` | Feature-bazli kredi hesaplama (Words, Images, Minutes, vb.) |
| `app/Domains/Entity/Enums/EntityEnum.php` | Model tanimlari, unitPrice, creditIndex |
| `app/Domains/Entity/BaseDriver.php` | Tum driver'larin base class'i |
| `app/Models/User.php` + `HasCredit.php` trait | Kullanici kredi alanlari |
| `app/Models/Plan.php` | Plan modeli, ai_models JSON yapisi |
| `app/Services/PaymentGateways/Contracts/CreditUpdater.php` | Odeme sonrasi kredi ekleme |
| `app/Models/UserUsageCredit.php` | Kredi kullanim gecmisi |

### 1.4 Kredi Turleri (Mevcut Calculate Traitler)

| Trait | Birim | Kullanan Featurelar |
|-------|-------|---------------------|
| `HasWords` | Kelime sayisi | Chat, Writer, Code |
| `HasImages` | Gorsel sayisi | DALL-E, SD, Flux |
| `HasTextToSpeech` | Voice sayisi | TTS (Google, OpenAI, ElevenLabs) |
| `HasSpeechToText` | Kelime sayisi | Whisper |
| `HasTextToVideo` | Video sayisi | Sora, Kling, Luma |
| `HasMinute` | Dakika | AI Music |
| `HasSecond` | Saniye | Sora (sure-bazli) |
| `HasCharacters` | Karakter sayisi | Bazi ozel modeller |
| `HasPlagiarism` | Kelime sayisi | Plagiarism check |

---

## 2. Shared Credit Pool Mimarisi

### 2.1 Temel Konsept

```
MEVCUT SISTEM (Ayri Krediler)          YENI SISTEM (Paylasilmis Havuz)
================================       ================================
GPT-4o:     1000 kredi                 Toplam Havuz: 10.000 kredi
DALL-E 3:    200 kredi                   |
Claude:      500 kredi                   |-- GPT-4o kullanimi:    1 kredi/kelime
Whisper:    2000 kredi                   |-- DALL-E 3 kullanimi: 50 kredi/gorsel
  (her model ayri)                       |-- Claude kullanimi:    2 kredi/kelime
                                         |-- Video kullanimi:   200 kredi/video
                                         |-- TTS kullanimi:      5 kredi/ses
                                         (tek havuzdan harcama)
```

### 2.2 Iki Sistem Birlikte Calisma Modeli

```
Kullanici istek yapar
  |
  v
Kullanicinin kredi sistemi ne?
  |
  |-- "separated" --> Mevcut entity_credits sistemi (degismez)
  |
  |-- "shared"    --> Shared credit pool sistemi (yeni)
       |
       v
     shared_credits bakiyesinden
     model maliyetine gore dus
```

**Karar Mekanizmasi:**
- Global ayar: `credit_system_mode` = `separated` | `shared` | `both`
- Plan bazli: Her plana `credit_system_type` alani eklenir
- Kullanici bazli: `users.credit_system_type` ile override edilebilir

### 2.3 Shared Credit Maliyet Hesaplama

Her model icin bir **shared credit cost** tanimlanir. Bu, adminin kontrol edebilecegi bir deger:

```
Shared Credit Cost = base_cost * quality_multiplier * quantity

Ornek:
- GPT-4o chat (1 kelime)     = 1 x 1.0 x 1 = 1 kredi
- GPT-4o chat (1 kelime, HD)  = 1 x 1.5 x 1 = 1.5 kredi
- DALL-E 3 (1 gorsel)        = 50 x 1.0 x 1 = 50 kredi
- DALL-E 3 (1 gorsel, HD)    = 50 x 2.0 x 1 = 100 kredi
- Sora video (1 adet, 4sn)   = 200 x 1.0 x 1 = 200 kredi
- Sora video (1 adet, 8sn)   = 200 x 2.0 x 1 = 400 kredi
```

---

## 3. Veritabani Degisiklikleri

### 3.1 Yeni Tablolar

#### `shared_credit_costs` - Model Bazli Maliyet Tablosu

```php
Schema::create('shared_credit_costs', function (Blueprint $table) {
    $table->id();
    $table->string('entity_key');           // EntityEnum slug: 'gpt-4o', 'dall-e-3'
    $table->string('engine_key');           // EngineEnum slug: 'openai', 'anthropic'
    $table->string('feature_type');         // 'word', 'image', 'video', 'tts', 'stt', 'minute', 'second'
    $table->decimal('base_cost', 10, 4)    // Birim basina temel maliyet
        ->default(1.0000);
    $table->decimal('quality_high_multiplier', 5, 2) // Yuksek kalite carpani
        ->default(1.50);
    $table->decimal('quality_low_multiplier', 5, 2)  // Dusuk kalite carpani
        ->default(0.75);
    $table->boolean('is_active')->default(true);
    $table->timestamps();

    $table->unique(['entity_key', 'engine_key']);
});
```

#### `shared_credit_transactions` - Kredi Islem Gecmisi

```php
Schema::create('shared_credit_transactions', function (Blueprint $table) {
    $table->id();
    $table->foreignId('user_id')->constrained()->cascadeOnDelete();
    $table->string('entity_key');           // Hangi model kullanildi
    $table->string('engine_key');           // Hangi engine
    $table->string('feature_type');         // word, image, video, vb.
    $table->string('action_type');          // 'deduct', 'topup', 'plan_assign', 'refund', 'expire', 'admin_adjust'
    $table->decimal('amount', 12, 4);      // Kredi miktari (negatif: harcama, pozitif: ekleme)
    $table->decimal('balance_after', 12, 4); // Islem sonrasi bakiye
    $table->decimal('unit_cost', 10, 4)->nullable();  // Birim maliyet (harcamalarda)
    $table->integer('quantity')->default(1);           // Miktar
    $table->string('quality')->nullable();             // 'low', 'standard', 'high'
    $table->json('metadata')->nullable();              // Ek bilgi (prompt ozeti, resim boyutu, vb.)
    $table->string('description')->nullable();         // Okunabilir aciklama
    $table->timestamps();

    $table->index(['user_id', 'created_at']);
    $table->index(['user_id', 'action_type']);
    $table->index(['entity_key', 'created_at']);
});
```

#### `shared_credit_packages` - Top-up Paketleri (Kredi Satin Alma)

```php
Schema::create('shared_credit_packages', function (Blueprint $table) {
    $table->id();
    $table->string('name');                 // 'Starter Pack', 'Pro Pack'
    $table->decimal('credits', 12, 4);     // Verilen kredi miktari
    $table->decimal('price', 10, 2);       // Fiyat (USD)
    $table->string('currency')->default('USD');
    $table->decimal('bonus_credits', 12, 4)->default(0); // Bonus kredi
    $table->boolean('is_active')->default(true);
    $table->boolean('is_featured')->default(false);
    $table->integer('sort_order')->default(0);
    $table->text('description')->nullable();
    $table->string('stripe_product_id')->nullable();
    $table->string('stripe_price_id')->nullable();
    $table->timestamps();
});
```

### 3.2 Mevcut Tablolara Eklenen Alanlar

#### `users` tablosu

```php
// Migration: add_shared_credits_to_users_table
$table->decimal('shared_credits', 12, 4)->default(0);     // Paylasilmis kredi bakiyesi
$table->string('credit_system_type')                        // 'separated' | 'shared'
    ->default('separated');
$table->timestamp('shared_credits_expires_at')->nullable(); // Kredi son kullanma tarihi (opsiyonel)
```

#### `plans` tablosu

```php
// Migration: add_shared_credit_fields_to_plans_table
$table->string('credit_system_type')                        // 'separated' | 'shared'
    ->default('separated');
$table->decimal('shared_credits_amount', 12, 4)            // Plan ile verilen shared kredi
    ->default(0);
$table->json('shared_credit_model_overrides')->nullable();  // Plan-bazli model maliyet override'lari
$table->json('shared_credit_feature_limits')->nullable();   // Ozellik bazli limit (opsiyonel)
```

#### `teams` tablosu

```php
// Migration: add_shared_credits_to_teams_table
$table->decimal('shared_credits', 12, 4)->default(0);
$table->string('credit_system_type')->default('separated');
```

### 3.3 Settings Tablosuna Eklenecek Ayarlar

```php
// Global ayarlar (settings tablosu uzerinden)
'shared_credit_system_enabled'     => false,           // Sistem aktif mi?
'shared_credit_default_base_cost'  => 1.0,             // Varsayilan birim maliyet
'shared_credit_show_cost_preview'  => true,             // Islem oncesi maliyet gosterilsin mi?
'shared_credit_allow_topup'        => true,             // Kullanicilar ek kredi alabilir mi?
'shared_credit_allow_negative'     => false,            // Negatif bakiye izin verilsin mi?
'shared_credit_expiration_days'    => 0,                // 0 = son kullanma yok
'shared_credit_migration_mode'     => 'manual',         // 'manual' | 'auto' | 'disabled'
```

---

## 4. Backend Implementasyonu

### 4.1 Yeni Service Katmani

#### `app/Services/SharedCredit/SharedCreditService.php`

```
SharedCreditService
├── getBalance(User $user): float
├── hasBalance(User $user, float $requiredAmount): bool
├── deduct(User $user, DeductionContext $context): SharedCreditTransaction
├── topUp(User $user, float $amount, string $reason): SharedCreditTransaction
├── refund(User $user, SharedCreditTransaction $original): SharedCreditTransaction
├── calculateCost(EntityEnum $entity, CalculationInput $input): CostEstimate
├── previewCost(EntityEnum $entity, CalculationInput $input): CostPreview
├── getTransactionHistory(User $user, ?Carbon $from, ?Carbon $to): Collection
├── expireCredits(): void  // Scheduled job
└── migrateUserCredits(User $user): void
```

#### `app/Services/SharedCredit/CostCalculator.php`

```
CostCalculator
├── calculate(EntityEnum $entity, string $featureType, int $quantity, ?string $quality): float
├── getBaseCost(EntityEnum $entity): float
├── getQualityMultiplier(EntityEnum $entity, string $quality): float
├── getModelOverrides(?Plan $plan): array
└── formatCostDisplay(float $cost): string
```

#### DTO'lar (Data Transfer Objects)

```php
// app/Services/SharedCredit/DTOs/DeductionContext.php
class DeductionContext {
    public function __construct(
        public EntityEnum $entity,
        public string $featureType,    // 'word', 'image', 'video', vb.
        public float $quantity,
        public ?string $quality = 'standard',
        public ?array $metadata = null,
    ) {}
}

// app/Services/SharedCredit/DTOs/CostEstimate.php
class CostEstimate {
    public function __construct(
        public float $totalCost,
        public float $baseCost,
        public float $qualityMultiplier,
        public float $quantity,
        public string $featureType,
        public bool $canAfford,
        public float $currentBalance,
    ) {}
}
```

### 4.2 HasCreditLimit Trait Modifikasyonu

`HasCreditLimit.php` dosyasindaki degisiklik **minimal ve geriye uyumlu** olmali:

```php
// app/Domains/Entity/Concerns/HasCreditLimit.php

// MEVCUT: creditBalance() metodu
public function creditBalance(): float
{
    // YENI: Shared credit sistemi kontrolu
    if ($this->isSharedCreditUser()) {
        return app(SharedCreditService::class)->getBalance($this->getUser());
    }

    // ... mevcut entity_credits mantigi aynen kalir ...
}

// MEVCUT: hasCreditBalance() metodu
public function hasCreditBalance(): bool
{
    if ($this->guest) {
        return $this->guestHasAttempts();
    }

    if ($this->isSharedCreditUser()) {
        return app(SharedCreditService::class)->hasBalance($this->getUser(), 0.01);
    }

    return $this->creditBalance() > 0 || $this->isUnlimitedCredit();
}

// MEVCUT: hasCreditBalanceForInput() metodu
public function hasCreditBalanceForInput(): bool
{
    if ($this->isSharedCreditUser()) {
        $cost = app(CostCalculator::class)->calculate(
            $this->enum(),
            $this->getFeatureType(),
            $this->getCalculatedInputCredit(),
            $this->getQuality()
        );
        return app(SharedCreditService::class)->hasBalance($this->getUser(), $cost);
    }

    // ... mevcut mantik aynen kalir ...
}

// MEVCUT: decreaseCredit() metodu
public function decreaseCredit(float $value = 1.00): bool
{
    if ($this->guest || $this->isUnlimitedCredit()) {
        return true;
    }

    if ($this->isSharedCreditUser()) {
        return $this->decreaseSharedCredit($value);
    }

    // ... mevcut entity_credits mantigi aynen kalir ...
}

// YENI: Shared credit dusme
private function decreaseSharedCredit(float $value): bool
{
    $context = new DeductionContext(
        entity: $this->enum(),
        featureType: $this->getFeatureType(),
        quantity: $value,
        quality: $this->getQuality(),
    );

    $transaction = app(SharedCreditService::class)->deduct($this->getUser(), $context);

    return $transaction !== null;
}

// YENI: Kullanicinin shared credit kullanip kullanmadigini kontrol et
private function isSharedCreditUser(): bool
{
    if (! setting('shared_credit_system_enabled', false)) {
        return false;
    }

    $user = $this->getUser();
    if (! $user) {
        return false;
    }

    return $user->credit_system_type === 'shared';
}
```

**Onemli:** Bu yaklasim mevcut kodu **hic bozmaz**. `isSharedCreditUser()` false dondugunde mevcut kod tamamen ayni calisir.

### 4.3 CreditUpdater Trait Modifikasyonu

Plan atama ve yenilemede shared credit desteği:

```php
// app/Services/PaymentGateways/Contracts/CreditUpdater.php

public function creditIncreaseSubscribePlan(User $user, Plan $plan): void
{
    // YENI: Shared credit sistemi
    if ($plan->credit_system_type === 'shared') {
        $this->assignSharedCredits($user, $plan);
        return;
    }

    // ... mevcut entity_credits mantigi aynen kalir ...
}

private function assignSharedCredits(User $user, Plan $plan): void
{
    $amount = $plan->shared_credits_amount;

    if ($plan->reset_credits_on_renewal) {
        $user->shared_credits = $amount;
    } else {
        $user->shared_credits += $amount;
    }

    $user->credit_system_type = 'shared';
    $user->save();

    // Islem kaydini olustur
    app(SharedCreditService::class)->topUp(
        $user,
        $amount,
        "Plan atamasi: {$plan->name}"
    );
}
```

### 4.4 Model Degisiklikleri

#### User Model

```php
// app/Models/User.php - fillable'a ekle:
'shared_credits',
'credit_system_type',
'shared_credits_expires_at',

// Cast ekle:
'shared_credits' => 'decimal:4',
'shared_credits_expires_at' => 'datetime',

// Yeni method:
public function isSharedCreditUser(): bool
{
    return $this->credit_system_type === 'shared';
}

public function getSharedCreditBalance(): float
{
    return (float) $this->shared_credits;
}
```

#### Plan Model

```php
// app/Models/Plan.php - fillable'a ekle:
'credit_system_type',
'shared_credits_amount',
'shared_credit_model_overrides',
'shared_credit_feature_limits',

// Cast ekle:
'shared_credits_amount' => 'decimal:4',
'shared_credit_model_overrides' => 'array',
'shared_credit_feature_limits' => 'array',
```

### 4.5 Yeni Modeller

```php
// app/Models/SharedCreditCost.php
// app/Models/SharedCreditTransaction.php
// app/Models/SharedCreditPackage.php
```

### 4.6 Atomic Kredi Dusme (Race Condition Koruması)

```php
// SharedCreditService::deduct() icinde
public function deduct(User $user, DeductionContext $context): ?SharedCreditTransaction
{
    $cost = app(CostCalculator::class)->calculate(
        $context->entity,
        $context->featureType,
        $context->quantity,
        $context->quality
    );

    // DB::transaction + pessimistic locking
    return DB::transaction(function () use ($user, $cost, $context) {
        // Pessimistic lock ile kullaniciyi kilitle
        $lockedUser = User::where('id', $user->id)->lockForUpdate()->first();

        if ($lockedUser->shared_credits < $cost) {
            return null; // Yetersiz bakiye
        }

        $lockedUser->shared_credits -= $cost;
        $lockedUser->save();

        return SharedCreditTransaction::create([
            'user_id'       => $user->id,
            'entity_key'    => $context->entity->slug(),
            'engine_key'    => $context->entity->engine()->slug(),
            'feature_type'  => $context->featureType,
            'action_type'   => 'deduct',
            'amount'        => -$cost,
            'balance_after' => $lockedUser->shared_credits,
            'unit_cost'     => $cost / max($context->quantity, 1),
            'quantity'      => $context->quantity,
            'quality'       => $context->quality,
            'metadata'      => $context->metadata,
        ]);
    });
}
```

### 4.7 API Endpointleri

```php
// routes/api.php veya routes/panel.php'ye eklenecek

// Kullanici tarafli
Route::middleware('auth')->prefix('shared-credits')->group(function () {
    Route::get('/balance', [SharedCreditController::class, 'balance']);
    Route::get('/history', [SharedCreditController::class, 'history']);
    Route::post('/preview-cost', [SharedCreditController::class, 'previewCost']);
    Route::get('/packages', [SharedCreditController::class, 'packages']);
    Route::post('/purchase', [SharedCreditController::class, 'purchase']);
});

// Admin tarafli
Route::middleware(['auth', 'admin'])->prefix('admin/shared-credits')->group(function () {
    Route::get('/costs', [AdminSharedCreditController::class, 'costs']);
    Route::put('/costs/{id}', [AdminSharedCreditController::class, 'updateCost']);
    Route::post('/costs/bulk-update', [AdminSharedCreditController::class, 'bulkUpdateCosts']);
    Route::post('/assign', [AdminSharedCreditController::class, 'assignCredits']);
    Route::post('/migrate-user', [AdminSharedCreditController::class, 'migrateUser']);
    Route::post('/migrate-all', [AdminSharedCreditController::class, 'migrateAll']);
    Route::get('/packages', [AdminSharedCreditPackageController::class, 'index']);
    Route::post('/packages', [AdminSharedCreditPackageController::class, 'store']);
    Route::put('/packages/{id}', [AdminSharedCreditPackageController::class, 'update']);
    Route::delete('/packages/{id}', [AdminSharedCreditPackageController::class, 'destroy']);
    Route::get('/analytics', [AdminSharedCreditController::class, 'analytics']);
});
```

---

## 5. Admin Panel Degisiklikleri

### 5.1 Global Ayarlar Sayfasi

**Konum:** Admin > Settings > Shared Credit System (yeni sekme/sayfa)

| Ayar | Tur | Aciklama |
|------|-----|----------|
| Sistemi Etkinlestir | Toggle | Shared credit sistemini ac/kapat |
| Varsayilan Birim Maliyet | Number | Yeni modeller icin varsayilan maliyet |
| Maliyet Onizleme Goster | Toggle | Kullaniciya islem oncesi maliyeti goster |
| Top-up Izin Ver | Toggle | Kullanicilarin ek kredi almasi |
| Negatif Bakiye | Toggle | Bakiye sifirin altina dusebilir mi |
| Kredi Son Kullanma (Gun) | Number | 0 = son kullanma yok |
| Migrasyon Modu | Select | manual / auto / disabled |

### 5.2 Model Maliyet Yonetim Sayfasi

**Konum:** Admin > Finance > Shared Credit Costs (yeni sayfa)

Tum AI modellerini listeleyen bir tablo:

| Model | Engine | Feature | Baz Maliyet | Yuksek Kalite (x) | Dusuk Kalite (x) | Durum |
|-------|--------|---------|-------------|--------------------|--------------------|-------|
| GPT-4o | OpenAI | Kelime | 1.00 | 1.50x | 0.75x | Aktif |
| GPT-4o-mini | OpenAI | Kelime | 0.30 | 1.50x | 0.75x | Aktif |
| DALL-E 3 | OpenAI | Gorsel | 50.00 | 2.00x | 0.50x | Aktif |
| Claude Opus | Anthropic | Kelime | 2.00 | 1.50x | 0.75x | Aktif |
| Sora | OpenAI | Saniye | 25.00 | 2.00x | 1.00x | Aktif |
| ElevenLabs TTS | ElevenLabs | Ses | 5.00 | 1.00x | 1.00x | Aktif |

Admin her satiri duzenleyebilir. `EntityEnum`'daki tum modeller otomatik listelenir, DB'deki override yoksa `unitPrice()` degerinden otomatik hesaplanir.

### 5.3 Plan Olusturma/Duzenleme Formu

Mevcut 4-adimli formun **Step 1**'ine yeni alan eklenir:

```
[Kredi Sistemi Tipi]
  ( ) Ayri Krediler (Mevcut)     -> Step 4'te model-bazli kredi atamasi gosterilir
  (o) Paylasilmis Kredi Havuzu   -> Step 4'te tek "Toplam Kredi" alani gosterilir
```

**Step 4 Degisikligi (credit_system_type = 'shared' ise):**

```
Toplam Shared Kredi: [____10000____]

(Opsiyonel) Model Maliyet Override:
+----------------------------------------+
| Model       | Baz Maliyet (Override)   |
|-------------|--------------------------|
| GPT-4o      | [1.00] (global: 1.00)    |
| DALL-E 3    | [40.00] (global: 50.00)  | <- Bu planda daha ucuz
+----------------------------------------+

(Opsiyonel) Ozellik Limitleri:
+----------------------------------+
| Ozellik     | Aylik Limit        |
|-------------|--------------------|
| Gorsel      | [100] / sinirsiz   |
| Video       | [10]  / sinirsiz   |
| Ses         | [50]  / sinirsiz   |
+----------------------------------+
```

### 5.4 Kullanici Yonetimi

**Admin > Users > Edit User** sayfasina eklenenler:

- Kullanicinin kredi sistemi tipi (separated/shared) - degistirilebilir
- Shared credit bakiyesi - duzenlenebilir
- Shared credit gecmisi - son islemler listesi
- "Kredi Ekle/Cikar" butonu
- "Shared Credit'e Migrasyon Yap" butonu

### 5.5 Toplu Migrasyon Sayfasi

**Konum:** Admin > Finance > Credit Migration (yeni sayfa)

- Tum kullanicilari listeler
- Filtered: "Sadece Separated Credit kullanicilar"
- "Secilileri Migrate Et" toplu islem
- "Tum Kullanicilari Migrate Et" (onay dialogu ile)
- Migrasyon onizleme: "Bu kullanicinin mevcut 15 model kredisi, tahmini X shared credit'e donusturulecek"

---

## 6. Frontend / UX Degisiklikleri

### 6.1 Kredi Bakiye Gosterimi

**Header/Navbar'da:**

```
Mevcut (separated):    "Words: 5,000 | Images: 200"
Yeni (shared):         "Credits: 8,450"  (veya progress bar ile)
```

Shared credit kullanicilari icin tek bir bakiye gosterimi. `x-data` Alpine componenti:

```javascript
Alpine.data('sharedCreditBalance', () => ({
    balance: 0,
    loading: true,

    async init() {
        await this.fetchBalance();
        // Livewire event dinle (her islem sonrasi guncelleme)
        window.addEventListener('shared-credit-updated', () => this.fetchBalance());
    },

    async fetchBalance() {
        const res = await fetch('/api/shared-credits/balance');
        const data = await res.json();
        this.balance = data.balance;
        this.loading = false;
    },

    get formattedBalance() {
        return new Intl.NumberFormat().format(Math.floor(this.balance));
    }
}));
```

### 6.2 Islem Oncesi Maliyet Onizleme

Her AI ozelliginde, islem butonu yaninda maliyet gosterimi:

```
[Gorsel Olustur]  ~50 kredi

[Video Olustur]   ~200 kredi
  Kalite: HD -> ~400 kredi

[Ses Olustur]     ~5 kredi

[Mesaj Gonder]    (Chat icin: islem sonrasi hesaplanir - onizleme yok)
```

**Istisnalar:**
- Chat/Writer: Yanit uzunlugu onceden bilinmez, maliyet **sonradan** gosterilir
- Diger featurelar: Onceden hesaplanabilir, islem oncesi gosterilir

### 6.3 Maliyet Onizleme Componenti

```blade
{{-- resources/views/default/components/shared-credit-cost-preview.blade.php --}}
@props([
    'entity' => null,        // EntityEnum slug
    'featureType' => null,   // 'image', 'video', vb.
    'quantity' => 1,
    'quality' => 'standard',
    'show' => 'isSharedCreditUser', // Alpine kosulusu
])

<div x-show="{{ $show }}" x-cloak
     x-data="sharedCreditCostPreview({
         entity: '{{ $entity }}',
         featureType: '{{ $featureType }}',
         quantity: {{ $quantity }},
         quality: '{{ $quality }}'
     })"
     class="flex items-center gap-1.5 text-2xs text-foreground/60">
    <x-tabler-coins class="size-3.5" />
    <span x-text="costText"></span>
</div>
```

### 6.4 Kullanici Kredi Gecmisi Sayfasi

**Konum:** Dashboard > Credit Usage (yeni sayfa veya mevcut usage sayfasina ek sekme)

| Tarih | Ozellik | Model | Miktar | Maliyet | Bakiye |
|-------|---------|-------|--------|---------|--------|
| 30 Mar 14:22 | Gorsel | DALL-E 3 | 2 adet | -100 | 8,350 |
| 30 Mar 14:15 | Chat | GPT-4o | 245 kelime | -245 | 8,450 |
| 30 Mar 13:00 | Video | Sora | 1 adet (8sn) | -400 | 8,695 |
| 29 Mar 10:00 | Top-up | - | - | +5,000 | 9,095 |

### 6.5 Top-up (Kredi Satin Alma) Sayfasi

Higgsfield ornegine benzer bir sayfa:

```
+------------------+  +------------------+  +------------------+
|  Starter Pack    |  |  Pro Pack ★      |  |  Enterprise      |
|                  |  |                  |  |                  |
|  5,000 kredi     |  |  15,000 kredi    |  |  50,000 kredi    |
|  +500 bonus      |  |  +3,000 bonus    |  |  +15,000 bonus   |
|                  |  |                  |  |                  |
|  $9.99           |  |  $24.99          |  |  $79.99          |
|                  |  |                  |  |                  |
|  [Satin Al]      |  |  [Satin Al]      |  |  [Satin Al]      |
+------------------+  +------------------+  +------------------+
```

---

## 7. Migrasyon Stratejisi

### 7.1 Mevcut Kullanicilarin Donusturulmesi

Mevcut kullanicilarin `entity_credits`'ini shared credit'e cevirme formulu:

```
shared_credits = SUM(
    her model icin: model_credit * model_shared_cost
)

Ornek:
  GPT-4o:     1000 kelime kredisi * 1.0 maliyet = 1,000 shared kredi
  DALL-E 3:    200 gorsel kredisi * 50.0 maliyet = 10,000 shared kredi
  Claude:      500 kelime kredisi * 2.0 maliyet  = 1,000 shared kredi
  -------------------------------------------------------
  TOPLAM:                                         12,000 shared kredi
```

### 7.2 Migrasyon Modlari

| Mod | Aciklama |
|-----|----------|
| `disabled` | Migrasyon yapilmaz, admin tek tek kullanici atar |
| `manual` | Admin panelden toplu veya tekli migrasyon baslatiir |
| `auto` | Yeni planlar otomatik shared, mevcut kullanicilar plan yenilemesinde migrate edilir |

### 7.3 Geri Donusturulebilirlik

Migrasyon geri alinabilir olmali:
- `shared_credit_transactions` tablosu `plan_assign` tipli kayit tutar
- Admin "Separated Credit'e Geri Don" yapabilir
- Eski `entity_credits` degerleri `metadata` alaninda yedeklenir

---

## 8. Guvenlik ve Performans

### 8.1 Race Condition Korunmasi

- `lockForUpdate()` ile pessimistic locking (Section 4.6)
- Tek bir `DB::transaction()` icinde bakiye kontrolu + dusme
- `shared_credits` alani `decimal(12,4)` - hassas hesaplama

### 8.2 Cache Stratejisi

```php
// YAPMA: Bakiyeyi cache'leme!
// Kredi bakiyesi her zaman DB'den okunmali (tutarlilik icin)

// YAP: Model maliyetlerini cache'le (nadiren degisir)
Cache::remember('shared_credit_costs', 3600, function () {
    return SharedCreditCost::all()->keyBy('entity_key');
});

// Cache invalidation: Admin maliyet degistirdiginde
SharedCreditCost::observe(SharedCreditCostObserver::class);
// -> observer'da Cache::forget('shared_credit_costs')
```

### 8.3 Double-Spending Onleme

1. **DB Transaction + Lock:** Her dusme atomik
2. **Idempotency Key:** Ayni istek iki kez islenmez
3. **Balance Check in Transaction:** Lock alininca bakiye tekrar kontrol edilir
4. **Negatif Bakiye Engeli:** `max(0, balance - cost)` + hata firlatma

### 8.4 Rate Limiting

- Kredi dusme istekleri rate-limit ile korunur
- Ani toplu islemlere karsi throttle

### 8.5 Audit Trail

- Her kredi hareketi `shared_credit_transactions` tablosuna kaydedilir
- `metadata` JSON alaninda detayli bilgi (prompt ozeti, parametreler)
- Admin islemleri admin user_id ile loglanir

---

## 9. Faz Plani ve Oncelikler

### Faz 1: Temel Altyapi (Oncelik: Kritik)

**Hedef:** Shared credit sistemi calistirmak

1. Veritabani migrasyonlari olustur (tum yeni tablolar + mevcut tablo degisiklikleri)
2. `SharedCreditCost`, `SharedCreditTransaction`, `SharedCreditPackage` modelleri
3. `SharedCreditService` ve `CostCalculator` servisleri
4. `HasCreditLimit.php` modifikasyonu (geriye uyumlu)
5. `CreditUpdater.php` modifikasyonu (plan atamada shared credit desteği)
6. Seeder: Tum mevcut `EntityEnum` modelleri icin varsayilan `shared_credit_costs` kayitlari
7. Unit testler

### Faz 2: Admin Panel (Oncelik: Yuksek)

1. Admin > Settings > Shared Credit System ayar sayfasi
2. Admin > Finance > Shared Credit Costs (model maliyet yonetimi)
3. Plan formu Step 1'e `credit_system_type` secimi
4. Plan formu Step 4'e shared credit modu (tek havuz + overrides)
5. Kullanici duzenleme sayfasinda shared credit alanlari
6. Feature testler

### Faz 3: Kullanici Arayuzu (Oncelik: Yuksek)

1. Navbar'da shared credit bakiye gosterimi
2. Maliyet onizleme componenti
3. Islem sonrasi bakiye guncelleme (Livewire/Alpine event)
4. Kullanici kredi gecmisi sayfasi
5. E2E testler

### Faz 4: Odeme ve Top-up (Oncelik: Orta)

1. `SharedCreditPackage` CRUD admin paneli
2. Top-up satin alma sayfasi (kullanici)
3. Payment gateway entegrasyonu (Stripe, PayPal, vb.)
4. Satin alma sonrasi kredi ekleme
5. Integration testler

### Faz 5: Migrasyon Araclari (Oncelik: Orta)

1. Tekli kullanici migrasyon komutu
2. Toplu migrasyon admin sayfasi
3. Otomatik migrasyon (plan yenilemesinde)
4. Geri donusturme araci
5. Migrasyon raporlari

### Faz 6: Gelismis Ozellikler (Oncelik: Dusuk)

1. Kredi son kullanma tarihi (expiration)
2. Plan-bazli ozellik limitleri (aylik X gorsel siniri)
3. Kredi kullanim analizleri (admin dashboard)
4. Webhook/notification: dusuk bakiye uyarisi
5. Team shared credit desteği

---

## 10. Riskler ve Cozum Onerileri

### 10.1 Teknik Riskler

| Risk | Olasilik | Etki | Cozum |
|------|----------|------|-------|
| Race condition (cift harcama) | Orta | Yuksek | DB transaction + pessimistic lock |
| Cache tutarsizligi | Yuksek | Yuksek | Bakiyeyi ASLA cache'leme, sadece maliyetleri cache'le |
| Extension uyumsuzlugu | Dusuk | Orta | Extension'lar `HasCreditLimit` trait'i kullandigi icin otomatik uyumlu |
| Performans (her istekte DB lock) | Orta | Dusuk | Lock sadece yazma isleminde, okuma lock'suz |
| Migrasyon veri kaybi | Dusuk | Yuksek | Eski verileri metadata'da yedekle, geri donusturme araci sun |

### 10.2 Is Riskleri

| Risk | Cozum |
|------|-------|
| Mevcut kullanicilar karisabilir | Varsayilan "separated" kal, admin isterse "shared" a gecirir |
| Yanlis fiyatlandirma | Varsayilan maliyetler `EntityEnum::unitPrice()`'dan turetilir, admin override edebilir |
| Kotu niyetli kullanim (ucuz modeli exploit) | Her modelin maliyeti ayri, admin izleyip ayarlayabilir |

### 10.3 UX Riskleri

| Risk | Cozum |
|------|-------|
| Chat maliyeti onceden bilinmez | "Tahmini maliyet" goster veya "Maliyet yanit sonrasi hesaplanir" uyarisi |
| Kullanici bakiyenin neye yetecegini anlamaz | "Bu bakiye ile yaklasik X gorsel veya Y mesaj" gosterimi |
| Kompleks fiyat tablosu | Basit ve anlasilir kategoriler (Dusuk/Orta/Yuksek maliyet) |

---

## Ek: Mimari Diyagrami

```
┌─────────────────────────────────────────────────────────────────┐
│                        KULLANICI ISTEGI                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               v
┌──────────────────────────────────────────────────────────────────┐
│                     Entity::driver($model)                       │
│                      HasCreditLimit trait                         │
│                                                                  │
│  ┌────────────────────┐    ┌─────────────────────────────────┐  │
│  │  isSharedCreditUser │───>│  credit_system_type == 'shared' │  │
│  │      check          │    │        (User model)             │  │
│  └─────────┬───────────┘    └─────────────────────────────────┘  │
│            │                                                      │
│    ┌───────┴───────┐                                             │
│    │               │                                             │
│    v               v                                             │
│  FALSE           TRUE                                            │
│    │               │                                             │
│    v               v                                             │
│ ┌──────────┐  ┌──────────────────────────┐                      │
│ │ MEVCUT   │  │    SharedCreditService    │                      │
│ │ SISTEM   │  │                          │                      │
│ │          │  │  CostCalculator          │                      │
│ │ entity_  │  │    └─ shared_credit_costs │                      │
│ │ credits  │  │    └─ plan overrides     │                      │
│ │ JSON     │  │                          │                      │
│ │          │  │  DB::transaction +        │                      │
│ │ (aynen   │  │  lockForUpdate           │                      │
│ │  kalir)  │  │    └─ users.shared_      │                      │
│ │          │  │       credits             │                      │
│ └──────────┘  │                          │                      │
│               │  SharedCreditTransaction │                      │
│               │    └─ audit log          │                      │
│               └──────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Ek: Varsayilan Model Maliyetleri (Seed Data)

Asagidaki tablo, her modelin `EntityEnum::unitPrice()` degerini baz alarak
otomatik hesaplanan varsayilan shared credit maliyetlerini gosterir.
Admin bunlari DB uzerinden override edebilir.

| Kategori | Model | Birim | Onerilen Shared Cost |
|----------|-------|-------|---------------------|
| **Chat (Ekonomik)** | GPT-4o-mini | Kelime | 0.3 |
| **Chat (Standart)** | GPT-4o | Kelime | 1.0 |
| **Chat (Premium)** | Claude Opus | Kelime | 2.0 |
| **Chat (Premium)** | GPT-o1 | Kelime | 3.0 |
| **Gorsel (Standart)** | DALL-E 3 | Gorsel | 50 |
| **Gorsel (Premium)** | Stable Diffusion 3 Ultra | Gorsel | 80 |
| **Gorsel (Ekonomik)** | Flux Schnell | Gorsel | 15 |
| **Video (Standart)** | Sora | Saniye | 25 |
| **Video (Premium)** | Kling Pro | Video | 200 |
| **Ses** | ElevenLabs TTS | Ses | 5 |
| **Ses** | OpenAI TTS-1 | Ses | 3 |
| **Transkripsiyon** | Whisper | Kelime | 0.01 |
| **Muzik** | ElevenLabs Music | Dakika | 30 |

---

> **Not:** Bu dokuman bir teknik plan taslagi olup, implementasyon surecinde
> guncellenmesi beklenmektedir. Her faz tamamlandiginda ilgili bolum guncellenir.

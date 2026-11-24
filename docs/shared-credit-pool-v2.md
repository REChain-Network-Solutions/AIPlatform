# Shared Credit Pool - Anlayis & Teknik Plan v2

> **Tarih:** 2026-04-01
> **Durum:** Taslak - Onay Bekleniyor
> **Referans:** [Higgsfield Pricing](https://higgsfield.ai/pricing)

---

## Bolum 1: Ne Yapmak Istiyorsun - Anladigim Sekliyle

### Ozet

Mevcut sistemde her AI modeli icin **ayri ayri kredi** tanimlaniyor (Separated Credits). Bir kullanici GPT-4o icin 1000 kelime kredisi, DALL-E 3 icin 200 gorsel kredisi alir — ve bu krediler birbirinden bagimsizdir. GPT-4o kredisi bittiginde, elinde 200 DALL-E kredisi olsa bile bunu chat icin kullanamiyor.

Sen, bunun yanina **ikinci bir sistem** eklemek istiyorsun: **Shared Credit Pool**. Bu sistemde kullanici tek bir kredi havuzuna sahip olur (ornegin 10,000 kredi) ve bu krediyi **istediği AI ozelliginde** harcar. Farkli ozellikler farkli miktarda kredi tuketir (orn: 1 gorsel = 50 kredi, 1 chat kelimesi = 1 kredi, 1 video = 200 kredi).

### Detayli Anlayisim

**1. Iki Sistem Paralel Calisacak**

- Mevcut "separated credits" sistemi **hic degismeyecek, bozulmayacak**
- Yeni "shared credits" sistemi **yanina** eklenecek
- Hangi kullanicinin hangisini kullanacagini admin belirleyecek
- Mevcut kullanicilar, planlar, akislar etkilenmeyecek

**2. Kredi Tuketim Mantigi**

Her AI modeli/ozelligi icin bir **birim maliyet** tanimlanacak. Tuketim su formulle hesaplanir:

```
Toplam Maliyet = birim_maliyet x kalite_carpani x miktar

Ornekler:
- GPT-4o ile 500 kelimelik chat    = 1.0 x 1 x 500    = 500 kredi
- DALL-E 3 ile 2 gorsel (HD)       = 50 x 2.0 x 2      = 200 kredi
- Sora ile 1 video (8sn)           = 25 x 2.0 x 1       = 50 kredi
- Claude Opus ile 1000 kelime      = 2.0 x 1 x 1000     = 2000 kredi
```

**3. Admin Kontrolu - Genis Yetki**

Admin sunlari yapabilecek:
- Model bazli birim maliyetleri DB uzerinden ayarlama/override etme
- Hangi kullanicinin hangi kredi sistemini kullandigini belirleme
- Feature/ozellik bazli fiyatlandirma
- Mevcut kullanicilari toplu veya tekli olarak yeni sisteme migrate etme
- AI provider/model'leri plan bazinda acma/kapama (mevcut yetenek, korunacak)
- Feature ve template kisitlamalarini plan bazinda yonetme (mevcut yetenek, korunacak)
- Shared credit sistemini tamamen acma/kapama
- Kullanicilara manuel kredi atama

**4. Kullanici Deneyimi**

- Kalan kredi bakiyesi gorunecek (tek sayi: "8,450 kredi")
- Islem oncesi maliyet gosterilecek ("Bu islem ~50 kredi tuketecek")
- Kredi kullanim gecmisi gorulecek
- Ek kredi satin alinabilecek (top-up)
- **Istisna:** Chat/Writer gibi yanit uzunlugu onceden bilinemeyen durumlarda maliyet islem sonrasi hesaplanir

**5. Guvenlik & Teknik Gereksinimler**

- Atomik kredi dusme (race condition korunmasi)
- Cache'den kaynakli yanlis bakiye veya cift harcama olmamasi
- Kredi islem loglamasi (audit trail)
- Mevcut tum extension'larla (Image, Video, Chat, Voice, vb.) uyumluluk

**6. Opsiyonel Ozellikler**

- Kredi son kullanma tarihi
- Plan bazli ozellik limitleri (ornegin ayda max 100 gorsel)

---

## Bolum 2: Mevcut Sistem Analizi (Codebase'den)

### 2.1 Mevcut Kredi Yapisi

Kullanicinin `entity_credits` JSON alani su yapida saklanir (`users` tablosu):

```json
{
  "open_ai": {
    "gpt_4_o": { "credit": 1000.0, "isUnlimited": false },
    "dall_e_3": { "credit": 200.0, "isUnlimited": false }
  },
  "anthropic": {
    "claude_3_opus": { "credit": 500.0, "isUnlimited": true }
  },
  "gemini": {
    "gemini_2_5_pro": { "credit": 300.0, "isUnlimited": false }
  }
}
```

- Her engine (openai, anthropic, gemini, vb.) bir ust grup
- Her model (gpt_4_o, claude_3_opus, vb.) icerisinde `credit` ve `isUnlimited` alanlari
- Ayni yapi `plans.ai_models` JSON alaninda da kullaniliyor (plan tanimlari icin)
- Ayni yapi `teams.entity_credits` JSON alaninda da var (takim kredileri)

### 2.2 Kredi Dusme Akisi

```
1. Feature kodu Entity::driver($entityEnum)->forUser($user) cagirir
2. Driver, HasCreditLimit trait'indeki hasCreditBalance() ile kredi kontrolu yapar
3. AI API cagrisi gerceklestirilir
4. input($output)->calculateCredit() ile tuketim hesaplanir
5. decreaseCredit() cagirilir:
   - isUnlimited ise dogrudan true doner
   - unitPrice hesaplanir (EntityEnum::unitPrice())
   - UserUsageCredit tablosuna log yazilir
   - Global total_spend ayari guncellenir
   - updateUserCredit() ile entity_credits JSON'i guncellenir
```

### 2.3 Kritik Dosya ve Siniflar

| Dosya | Aciklama |
|-------|----------|
| `app/Domains/Entity/Concerns/HasCreditLimit.php` | Ana kredi mantigi: balance sorgulama, kredi dusme, artirma, set etme. **Shared credit entegrasyon noktasi.** |
| `app/Domains/Entity/Contracts/WithCreditInterface.php` | Kredi interface'i: `creditBalance()`, `decreaseCredit()`, `hasCreditBalance()`, vb. |
| `app/Domains/Entity/Enums/EntityEnum.php` | Tum AI modelleri, `unitPrice()` fiyatlari, `creditBy()`, `creditIndex()`, hesaplama trait'leri |
| `app/Domains/Entity/EntityStats.php` | `word()`, `image()`, `textToSpeech()` vb. kategorik kredi istatistikleri |
| `app/Models/Concerns/User/HasCredit.php` | User modeline eklenen trait: `getCredit()`, `getFreshCredits()`, `updateCredits()` |
| `app/Models/User.php` | `entity_credits` JSON alani, `HasCredit` trait'i |
| `app/Models/Plan.php` | `ai_models` JSON alani, `getCredit()`, plan tipleri (subscription/token_pack) |
| `app/Models/Team/Team.php` | `entity_credits` JSON alani (takim kredileri, User ile ayni yapi) |
| `app/Models/UserUsageCredit.php` | Her kredi dusme islemi icin audit log tablosu |
| `app/Services/PaymentGateways/Contracts/CreditUpdater.php` | Plan atama/iptal sirasinda kredi ekleme/cikarma: `creditIncreaseSubscribePlan()`, `creditDecreaseCancelPlan()` |
| `app/Livewire/AssignViewCredits.php` | Admin paneldeki plan kredi atama Livewire componenti |
| `app/Livewire/Admin/Finance/Plan/SubscriptionPlanCreate.php` | 4 adimli plan olusturma formu |
| `app/Livewire/Admin/Finance/Plan/TokenPackPlanCreate.php` | Token pack plan olusturma |
| `app/Http/Controllers/Finance/PaymentProcessController.php` | Odeme, abonelik, prepaid islemleri, admin plan atama |
| `resources/views/default/components/credit-list.blade.php` | Kullaniciya gosterilen kredi listesi componenti |
| `resources/views/livewire/assign-view-credits.blade.php` | Admin panelde model bazli kredi atama formu |

### 2.4 Mevcut Kredi Hesaplama Trait'leri

Her AI modeli bir veya daha fazla hesaplama trait'i implement eder:

| Trait | Birim | Aciklama | Ornekler |
|-------|-------|----------|----------|
| `WithWordsInterface` | Kelime | Chat, Writer, Code | GPT-4o, Claude, Gemini |
| `WithImagesInterface` | Gorsel | Image generation | DALL-E 3, Stable Diffusion, Flux |
| `WithTextToSpeechInterface` | Ses | TTS | ElevenLabs, OpenAI TTS |
| `WithSpeechToTextInterface` | Kelime | STT | Whisper |
| `WithTextToVideoInterface` | Video | Text-to-video | Sora, Kling |
| `WithImageToVideoInterface` | Video | Image-to-video | Runway, Kling |
| `WithMinuteInterface` | Dakika | AI Music | ElevenLabs Music |
| `WithSecondInterface` | Saniye | Sure-bazli video | Sora (sure) |
| `WithCharsInterface` | Karakter | Karakter-bazli | Ozel modeller |
| `WithPlagiarismInterface` | Kelime | Plagiarism check | Plagiarism API |

### 2.5 Mevcut Odeme ve Plan Yapisi

**Plan Tipleri:**
- `subscription` - Aylik/yillik yenilenen abonelik
- `token_pack` - Tek seferlik prepaid kredi paketi

**Abonelik Akisi:**
1. Kullanici plan secer -> Gateway'e (Stripe/PayPal/Iyzico/vb.) yonlendirilir
2. Basarili odeme -> `creditIncreaseSubscribePlan()` cagirilir
3. Plan'daki `ai_models` JSON'i kullanicinin `entity_credits`'ine aktarilir
4. `reset_credits_on_renewal` true ise eski kredit sifirlanip yenisi set edilir, false ise ustune eklenir

**Admin Plan Atama:**
- `PaymentProcessController::assignPlanByAdmin()` ile dogrudan plan atanabilir
- `PaymentProcessController::assignTokenByAdmin()` ile token pack atanabilir

**Desteklenen Gateway'ler:**
- Stripe, PayPal, Iyzico, Paystack, Razorpay, Bank Transfer, Free ve daha fazlasi
- Her gateway icin ayri Listener sinifi (`StripeWebhookListener`, `PaypalWebhookListener`, vb.)
- Hepsi `CreditUpdater` trait'ini kullaniyor

### 2.6 Extension Uyumlulugu

Tum extension'lar (PhotoStudio, AiAvatar, FashionStudio, Chatbot, ChatbotVoice, vb.) ayni `Entity::driver()->forUser()` pattern'ini kullaniyor. `HasCreditLimit` trait'i uzerinden kredi kontrolu ve dusme yapiyorlar. Bu da demek oluyor ki **`HasCreditLimit`'e yapilacak shared credit entegrasyonu otomatik olarak tum extension'lara yansiyacak.**

---

## Bolum 3: Teknik Plan

### 3.1 Mimari Yaklasim: Minimal Invaziv Strateji

Mevcut sisteme minimum mudahale ile yeni sistemi entegre etmek icin **Strategy Pattern** benzeri bir yaklasim:

```
HasCreditLimit::decreaseCredit()
  |
  |-- isSharedCreditUser()?
  |     |
  |     |-- HAYIR --> Mevcut entity_credits mantigi (DEGISMEZ)
  |     |
  |     |-- EVET  --> SharedCreditService::deduct() (YENI)
```

**Neden bu yaklasim?**
- `HasCreditLimit` trait'i TUM driver'larin kredi operasyonlari icin kullandigi merkezi nokta
- Buraya bir `if` eklemek, mevcut kodu **hic bozmadan** tum feature'lara shared credit desteği verir
- Extension'lar dahil hicbir sey degistirmeye gerek yok

### 3.2 Veritabani Degisiklikleri

#### Yeni Tablolar

**1. `shared_credit_costs` - Model Bazli Birim Maliyetler**

```php
Schema::create('shared_credit_costs', function (Blueprint $table) {
    $table->id();
    $table->string('entity_key')->unique();    // EntityEnum slug: 'gpt_4_o', 'dall_e_3'
    $table->string('engine_key');               // EngineEnum slug: 'open_ai', 'anthropic'
    $table->string('feature_type');             // 'word', 'image', 'video', 'tts', 'stt', 'minute', 'second', 'char'
    $table->decimal('base_cost', 10, 4)->default(1.0);
    $table->decimal('quality_high_multiplier', 5, 2)->default(1.50);
    $table->decimal('quality_low_multiplier', 5, 2)->default(0.75);
    $table->boolean('is_active')->default(true);
    $table->timestamps();
});
```

Seeder ile tum `EntityEnum` case'leri icin `unitPrice()` bazli varsayilan satirlar olusturulur.

**2. `shared_credit_transactions` - Islem Gecmisi (Audit Trail)**

```php
Schema::create('shared_credit_transactions', function (Blueprint $table) {
    $table->id();
    $table->foreignId('user_id')->constrained()->cascadeOnDelete();
    $table->string('entity_key')->nullable();   // Hangi model
    $table->string('engine_key')->nullable();   // Hangi engine
    $table->string('feature_type')->nullable(); // word, image, video, vb.
    $table->string('action_type');              // 'deduct', 'topup', 'plan_assign', 'refund', 'expire', 'admin_adjust'
    $table->decimal('amount', 12, 4);           // Negatif: harcama, Pozitif: ekleme
    $table->decimal('balance_after', 12, 4);    // Islem sonrasi bakiye
    $table->decimal('unit_cost', 10, 4)->nullable();
    $table->decimal('quantity', 12, 4)->default(1);
    $table->string('quality')->nullable();      // 'low', 'standard', 'high'
    $table->json('metadata')->nullable();       // Ek bilgi
    $table->string('description')->nullable();
    $table->timestamps();

    $table->index(['user_id', 'created_at']);
    $table->index(['user_id', 'action_type']);
});
```

**3. `shared_credit_packages` - Top-up Paketleri**

```php
Schema::create('shared_credit_packages', function (Blueprint $table) {
    $table->id();
    $table->string('name');
    $table->decimal('credits', 12, 4);
    $table->decimal('price', 10, 2);
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

#### Mevcut Tablolara Eklenen Alanlar

**`users` tablosu:**
```php
$table->decimal('shared_credits', 12, 4)->default(0);
$table->string('credit_system_type')->default('separated'); // 'separated' | 'shared'
$table->timestamp('shared_credits_expires_at')->nullable();
```

**`plans` tablosu:**
```php
$table->string('credit_system_type')->default('separated'); // 'separated' | 'shared'
$table->decimal('shared_credits_amount', 12, 4)->default(0);
$table->json('shared_credit_model_overrides')->nullable();   // Plan-bazli model maliyet override
$table->json('shared_credit_feature_limits')->nullable();    // Opsiyonel ozellik limitleri
```

**`teams` tablosu:**
```php
$table->decimal('shared_credits', 12, 4)->default(0);
$table->string('credit_system_type')->default('separated');
```

#### Settings Ayarlari

```php
'shared_credit_system_enabled'    => false,    // Sistem acik mi?
'shared_credit_default_base_cost' => 1.0,      // Varsayilan birim maliyet
'shared_credit_show_cost_preview' => true,      // Islem oncesi maliyet goster
'shared_credit_allow_topup'       => true,      // Top-up izin ver
'shared_credit_allow_negative'    => false,     // Negatif bakiye izin ver
'shared_credit_expiration_days'   => 0,         // 0 = son kullanma yok
'shared_credit_migration_mode'    => 'manual',  // 'manual' | 'auto' | 'disabled'
```

### 3.3 Backend: Yeni Service Katmani

#### `app/Services/SharedCredit/SharedCreditService.php`

Tum shared credit islemlerinin merkezi servisi:

```
SharedCreditService
├── getBalance(User $user): float
├── hasBalance(User $user, float $required): bool
├── deduct(User $user, DeductionContext $ctx): SharedCreditTransaction
│     └── DB::transaction + lockForUpdate (atomik islem)
├── topUp(User $user, float $amount, string $reason): SharedCreditTransaction
├── refund(User $user, SharedCreditTransaction $original): SharedCreditTransaction
├── calculateCost(EntityEnum $entity, string $featureType, float $qty, ?string $quality): float
├── previewCost(EntityEnum $entity, string $featureType, float $qty, ?string $quality): CostEstimate
├── getHistory(User $user, ?Carbon $from, ?Carbon $to): Collection
├── expireCredits(): void  (Scheduled command)
└── migrateFromSeparated(User $user): void
```

#### `app/Services/SharedCredit/CostCalculator.php`

Model bazli maliyet hesaplama:

```
CostCalculator
├── calculate(EntityEnum $entity, string $featureType, float $qty, ?string $quality): float
│     1. shared_credit_costs tablosundan base_cost al (cache'li)
│     2. Plan-bazli override varsa uygula
│     3. quality multiplier uygula
│     4. qty ile carp
│     5. Toplam maliyeti dondur
├── getBaseCost(EntityEnum $entity): float
├── getQualityMultiplier(EntityEnum $entity, string $quality): float
└── formatCostDisplay(float $cost): string
```

**Cache Stratejisi:**
- `shared_credit_costs` tablosu 1 saat cache'lenir (nadiren degisir)
- Admin maliyet degistirdiginde cache invalidate edilir
- **Kullanici bakiyesi ASLA cache'lenmez** (tutarlilik icin her zaman DB'den okunur)

#### DTO'lar

```php
// app/Services/SharedCredit/DTOs/DeductionContext.php
class DeductionContext {
    public function __construct(
        public EntityEnum $entity,
        public string $featureType,
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
        public bool $canAfford,
        public float $currentBalance,
    ) {}
}
```

### 3.4 HasCreditLimit Modifikasyonu (Entegrasyon Noktasi)

Bu trait mevcut sistemin kalbi. Degisiklik **minimal ve geriye uyumlu** olacak.

**Eklenen private method:**

```php
private function isSharedCreditUser(): bool
{
    if (! setting('shared_credit_system_enabled', false)) {
        return false;
    }
    $user = $this->getUser();
    return $user && $user->credit_system_type === 'shared';
}
```

**Degistirilen methodlar (sadece basa `if` ekleniyor):**

| Method | Degisiklik |
|--------|-----------|
| `creditBalance()` | Shared ise `SharedCreditService::getBalance()` dondur |
| `hasCreditBalance()` | Shared ise `SharedCreditService::hasBalance()` kontrol et |
| `hasCreditBalanceForInput()` | Shared ise maliyeti hesapla, yeterli mi kontrol et |
| `decreaseCredit()` | Shared ise `SharedCreditService::deduct()` cagir |
| `isUnlimitedCredit()` | Shared icin `shared_credits` sinirsiz olabilir kontrolu |
| `increaseCredit()` | Shared ise `SharedCreditService::topUp()` cagir |
| `setCredit()` | Shared ise `shared_credits` alanini set et |

**Her method'un mevcut mantigi tamamen korunur.** Sadece basina:

```php
if ($this->isSharedCreditUser()) {
    return /* shared credit logic */;
}
// ... mevcut kod aynen devam eder ...
```

### 3.5 CreditUpdater Modifikasyonu

`creditIncreaseSubscribePlan()` ve `creditDecreaseCancelPlan()` methodlarina plan tipi kontrolu eklenir:

```php
public static function creditIncreaseSubscribePlan(?User $user, ?Plan $plan): void
{
    if ($plan?->credit_system_type === 'shared') {
        // Shared credit ekle
        $amount = $plan->shared_credits_amount;
        if ($plan->reset_credits_on_renewal) {
            $user->shared_credits = $amount;
        } else {
            $user->shared_credits += $amount;
        }
        $user->credit_system_type = 'shared';
        $user->save();

        app(SharedCreditService::class)->topUp($user, $amount, "Plan: {$plan->name}");
        return;
    }

    // ... mevcut entity_credits mantigi aynen kalir ...
}
```

### 3.6 Race Condition Korunmasi (Atomik Dusme)

```php
// SharedCreditService::deduct()
public function deduct(User $user, DeductionContext $ctx): ?SharedCreditTransaction
{
    $cost = app(CostCalculator::class)->calculate(
        $ctx->entity, $ctx->featureType, $ctx->quantity, $ctx->quality
    );

    return DB::transaction(function () use ($user, $cost, $ctx) {
        // Pessimistic lock: baska islem ayni anda bakiyeyi degistiremesin
        $locked = User::where('id', $user->id)->lockForUpdate()->first();

        if ($locked->shared_credits < $cost && !setting('shared_credit_allow_negative')) {
            return null; // Yetersiz bakiye
        }

        $locked->shared_credits = max(0, $locked->shared_credits - $cost);
        $locked->save();

        return SharedCreditTransaction::create([
            'user_id'       => $user->id,
            'entity_key'    => $ctx->entity->slug(),
            'engine_key'    => $ctx->entity->engine()->slug(),
            'feature_type'  => $ctx->featureType,
            'action_type'   => 'deduct',
            'amount'        => -$cost,
            'balance_after' => $locked->shared_credits,
            'unit_cost'     => $cost / max($ctx->quantity, 0.01),
            'quantity'      => $ctx->quantity,
            'quality'       => $ctx->quality,
            'metadata'      => $ctx->metadata,
        ]);
    });
}
```

### 3.7 Model Degisiklikleri

**User Model (`app/Models/User.php`):**

```php
// fillable'a ekle:
'shared_credits', 'credit_system_type', 'shared_credits_expires_at'

// $casts'e ekle:
'shared_credits' => 'decimal:4',
'shared_credits_expires_at' => 'datetime',

// Yeni methodlar:
public function isSharedCreditUser(): bool
{
    return $this->credit_system_type === 'shared';
}
```

**Plan Model (`app/Models/Plan.php`):**

```php
// fillable'a ekle:
'credit_system_type', 'shared_credits_amount',
'shared_credit_model_overrides', 'shared_credit_feature_limits'

// $casts'e ekle:
'shared_credits_amount' => 'decimal:4',
'shared_credit_model_overrides' => 'array',
'shared_credit_feature_limits' => 'array',
```

**Yeni Modeller:**
- `app/Models/SharedCreditCost.php`
- `app/Models/SharedCreditTransaction.php`
- `app/Models/SharedCreditPackage.php`

### 3.8 Admin Panel Degisiklikleri

#### Global Ayarlar

Admin > Settings'e "Shared Credit System" sekmesi/sayfasi eklenir. Tum settings ayarlari buradan yonetilir.

#### Model Maliyet Yonetimi

Admin > Finance > "Shared Credit Costs" sayfasi. EntityEnum'daki tum modeller tabloda listelenir. Admin `base_cost`, `quality_high_multiplier`, `quality_low_multiplier` ve `is_active` degerlerini duzenleyebilir.

DB'de override yoksa `EntityEnum::unitPrice()` bazli otomatik hesaplanan deger kullanilir.

#### Plan Formuna Eklenenler

**Step 1'e:**
- `credit_system_type` radio secimi: "Ayri Krediler (Mevcut)" | "Paylasilmis Kredi Havuzu"

**Step 4'e (credit_system_type = 'shared' oldugunda):**
- Mevcut model-bazli kredi formu gizlenir
- Yerine tek bir "Toplam Shared Kredi" alani gosterilir
- Opsiyonel: Plan-bazli model maliyet override tablosu
- Opsiyonel: Feature bazli aylik limit tablosu

#### Kullanici Yonetimi

Admin > Users > Edit sayfasina:
- Kredi sistemi tipi (separated/shared) secimi
- Shared credit bakiyesi goruntuleme/duzenleme
- Son islemler listesi
- "Kredi Ekle", "Shared'e Migrate Et" butonlari

### 3.9 Frontend / UX

#### Navbar Kredi Gosterimi

Shared credit kullanicilari icin `x-credit-list` componenti adapte edilir:

```
Mevcut (separated): "Words: 5,000 | Images: 200"
Yeni (shared):      "Credits: 8,450" (tek progress bar)
```

`credit-list.blade.php`'de `isSharedCreditUser` kontrolu eklenir.

#### Maliyet Onizleme

Yeni Blade componenti: `<x-shared-credit-cost-preview>`

```blade
<x-shared-credit-cost-preview
    entity="dall_e_3"
    feature-type="image"
    :quantity="1"
    quality="standard"
/>
{{-- Cikti: "~50 kredi" --}}
```

Bu component sadece shared credit kullanicilarina gosterilir.

#### Islem Sonrasi Bakiye Guncelleme

`decreaseCredit()` tamamlandiginda bir Livewire/browser event dispatch edilir:

```javascript
window.dispatchEvent(new CustomEvent('shared-credit-updated', {
    detail: { balance: newBalance }
}));
```

Navbar'daki bakiye componenti bu event'i dinleyerek anlik guncellenir.

#### Kredi Gecmisi Sayfasi

Dashboard'a "Kredi Kullanimim" sayfasi eklenir. `shared_credit_transactions` tablosundan kullanicinin islemleri listelenir.

#### Top-up Sayfasi

`shared_credit_packages` tablosundaki aktif paketler kartlar halinde gosterilir. Mevcut payment gateway altyapisi kullanilarak satin alma yapilir.

### 3.10 Migrasyon Stratejisi

**Formul:**
```
shared_credits = SUM(model_credit * model_shared_cost) icin tum modeller
```

**Modlar:**
| Mod | Aciklama |
|-----|----------|
| `disabled` | Migrasyon yok, admin tek tek atar |
| `manual` | Admin panelden toplu/tekli migrasyon |
| `auto` | Plan yenilemesinde otomatik migrate |

**Geri Donusturulebilirlik:**
- Migrasyon sirasinda eski `entity_credits` degeri `shared_credit_transactions.metadata`'da yedeklenir
- Admin "Separated'a Geri Don" yapabilir

### 3.11 API Endpointleri

**Kullanici:**
```
GET    /shared-credits/balance       - Bakiye sorgulama
GET    /shared-credits/history       - Islem gecmisi
POST   /shared-credits/preview-cost  - Maliyet onizleme
GET    /shared-credits/packages      - Top-up paketleri
POST   /shared-credits/purchase      - Paket satin alma
```

**Admin:**
```
GET    /admin/shared-credits/costs           - Tum model maliyetleri
PUT    /admin/shared-credits/costs/{id}      - Maliyet guncelle
POST   /admin/shared-credits/costs/bulk      - Toplu guncelleme
POST   /admin/shared-credits/assign          - Kullaniciya kredi ata
POST   /admin/shared-credits/migrate-user    - Tekli migrasyon
POST   /admin/shared-credits/migrate-all     - Toplu migrasyon
CRUD   /admin/shared-credits/packages        - Top-up paket yonetimi
GET    /admin/shared-credits/analytics       - Kullanim analizleri
```

---

## Bolum 4: Faz Plani

### Faz 1: Temel Altyapi

1. Veritabani migrasyonlari (3 yeni tablo + 3 mevcut tablo degisikligi + settings)
2. Yeni modeller: `SharedCreditCost`, `SharedCreditTransaction`, `SharedCreditPackage`
3. Service katmani: `SharedCreditService`, `CostCalculator`, DTO'lar
4. `HasCreditLimit.php` modifikasyonu (if-guard ile shared credit yonlendirmesi)
5. `CreditUpdater.php` modifikasyonu (plan atamada shared credit desteği)
6. Seeder: Tum EntityEnum modelleri icin varsayilan `shared_credit_costs` kayitlari
7. Pest testleri

### Faz 2: Admin Panel

1. Settings sayfasi (shared credit system toggle + ayarlar)
2. Model maliyet yonetim sayfasi
3. Plan formuna `credit_system_type` + shared credit alanlari ekleme
4. Kullanici yonetiminde shared credit alanlari
5. Pest testleri

### Faz 3: Kullanici Arayuzu

1. Navbar'da shared credit bakiye gosterimi (credit-list componenti adaptasyonu)
2. Maliyet onizleme componenti
3. Islem sonrasi bakiye event'i
4. Kullanici kredi gecmisi sayfasi
5. E2E testler

### Faz 4: Top-up & Odeme

1. `SharedCreditPackage` admin CRUD
2. Top-up satin alma sayfasi (kullanici)
3. Mevcut payment gateway entegrasyonu
4. Integration testler

### Faz 5: Migrasyon Araclari

1. Tekli/toplu migrasyon admin paneli
2. Otomatik migrasyon (plan yenilemesinde)
3. Geri donusturme

### Faz 6: Gelismis (Opsiyonel)

1. Kredi son kullanma tarihi
2. Plan bazli ozellik limitleri
3. Dusuk bakiye uyari notification'i
4. Team shared credit desteği
5. Analizler dashboard'u

---

## Bolum 5: Mimari Diyagram

```
┌─────────────────────────────────────────────────────────────┐
│                    KULLANICI ISTEGI                           │
│   (Chat, Image, Video, Voice, Code, vb.)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           v
┌──────────────────────────────────────────────────────────────┐
│           Entity::driver($entityEnum)->forUser($user)         │
│                                                               │
│    ┌─────────────────────────────────────────────────────┐   │
│    │            HasCreditLimit trait                       │   │
│    │                                                     │   │
│    │   isSharedCreditUser()?                              │   │
│    │         │                                           │   │
│    │    ┌────┴────┐                                      │   │
│    │    │         │                                      │   │
│    │   NO        YES                                     │   │
│    │    │         │                                      │   │
│    │    v         v                                      │   │
│    │  ┌────────┐ ┌──────────────────────────────┐       │   │
│    │  │MEVCUT  │ │   SharedCreditService         │       │   │
│    │  │SISTEM  │ │                               │       │   │
│    │  │        │ │  CostCalculator               │       │   │
│    │  │entity_ │ │   └─ shared_credit_costs (DB) │       │   │
│    │  │credits │ │   └─ plan overrides           │       │   │
│    │  │JSON    │ │                               │       │   │
│    │  │        │ │  DB::transaction               │       │   │
│    │  │(aynen  │ │   + lockForUpdate()            │       │   │
│    │  │ kalir) │ │   └─ users.shared_credits      │       │   │
│    │  │        │ │                               │       │   │
│    │  │User    │ │  SharedCreditTransaction       │       │   │
│    │  │Usage   │ │   └─ audit log                 │       │   │
│    │  │Credit  │ │                               │       │   │
│    │  │(log)   │ │  UserUsageCredit               │       │   │
│    │  └────────┘ │   └─ (ayni log da yazilir)     │       │   │
│    │             └──────────────────────────────────┘       │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                               │
│    Extension'lar (PhotoStudio, Chatbot, Voice, vb.)           │
│    Ayni Entity::driver() pattern'ini kullandigi icin           │
│    otomatik olarak shared credit ile uyumlu calisir.           │
└──────────────────────────────────────────────────────────────┘
```

---

## Bolum 6: Riskler ve Cozumler

| Risk | Etki | Cozum |
|------|------|-------|
| Race condition (cift harcama) | Yuksek | DB transaction + pessimistic lock (`lockForUpdate`) |
| Cache tutarsizligi (yanlis bakiye) | Yuksek | Bakiyeyi ASLA cache'leme, sadece maliyetleri cache'le |
| Mevcut sistem kirilmasi | Kritik | `isSharedCreditUser()` false = mevcut kod degismez |
| Extension uyumsuzlugu | Dusuk | `HasCreditLimit` uzerinden entegrasyon, extension'lar etkilenmez |
| Migrasyon veri kaybi | Orta | Eski krediler metadata'da yedeklenir, geri donusum mumkun |
| Yanlis fiyatlandirma | Orta | Varsayilan degerler `unitPrice()` bazli, admin override edebilir |
| Chat maliyetini onceden gosterememe | Dusuk | "Maliyet yanit sonrasi hesaplanir" uyarisi goster |

---

## Bolum 7: creditIndex Optimizasyonu (Kilit Mimari Karar)

### 7.1 Kesfedilen Firsat

`EntityEnum::creditIndex()` methodu su an tum modeller icin sabit `1.0` donuyor. Ancak bu method **13 calculate trait** tarafindan dogrudan carpan olarak kullaniliyor:

```php
// HasWords.php
public function calculate(): float
{
    $wordCount = count(preg_split('/\PL+/u', $this->getInput(), -1, PREG_SPLIT_NO_EMPTY));
    return $wordCount * $this->getCreditIndex();
}
```

Ayni pattern tum trait'lerde tekrarlaniyor:

| Calculate Trait | Hesaplama Formulu |
|-----------------|-------------------|
| `HasWords` | `wordCount * getCreditIndex()` |
| `HasImages` | `imageCount * getCreditIndex()` |
| `HasTextToSpeech` | `voiceCount * getCreditIndex()` |
| `HasSpeechToText` | `wordCount * getCreditIndex()` |
| `HasTextToVideo` | `videoCount * getCreditIndex()` |
| `HasImageToVideo` | `videoCount * getCreditIndex()` |
| `HasVideoToVideo` | `videoCount * getCreditIndex()` |
| `HasSeconds` | `second * getCreditIndex()` |
| `HasMinutes` | `minute * getCreditIndex()` |
| `HasCharacters` | `charCount * getCreditIndex()` |
| `HasPresentation` | `presentationCount * getCreditIndex()` |
| `HasPlagiarism` | `wordCount * getCreditIndex()` |
| `HasVisionPreview` | `wordCount * getCreditIndex()` |

`getCreditIndex()` methodu `HasCreditLimit` trait'indedir ve `EntityEnum::creditIndex()` degerini dondurur:

```php
// HasCreditLimit.php (satir 353-356)
public function getCreditIndex(): float
{
    return $this->creditEnum()->creditIndex();
}
```

### 7.2 Neden Dogrudan creditIndex() Degistirilemez

`creditIndex()` degerini model bazli farkli yaparsak **mevcut separated sistem de etkilenir**:

```
SIMDI:    100 kelime GPT-4o → 100 * 1.0 = 100 kredi duser (entity_credits'ten)
DEGISIRSE: 100 kelime GPT-4o → 100 * 2.0 = 200 kredi duser (MEVCUT PLANLARI BOZAR!)
```

Bu yuzden `creditIndex()` **dokunulmaz**, yerine yeni bir method eklenir.

### 7.3 Cozum: sharedCreditIndex() + Akilli getCreditIndex()

#### Adim 1: EntityEnum'a `sharedCreditIndex()` Eklenir

```php
// app/Domains/Entity/Enums/EntityEnum.php

/** @noinspection PhpDuplicateMatchArmBodyInspection */
public function sharedCreditIndex(): float
{
    return match ($this) {
        // =============================================
        // CHAT MODELLERI - Kelime bazli (WithWordsInterface)
        // =============================================

        // Ekonomik (dusuk maliyet)
        self::GPT_4_O_MINI,
        self::GPT_4_1_MINI,
        self::GPT_4_1_NANO,
        self::GPT_3_5_TURBO,
        self::GPT_3_5_TURBO_0125,
        self::GPT_3_5_TURBO_1106,
        self::GPT_5_MINI,
        self::GPT_5_NANO            => 0.3,

        self::CLAUDE_3_HAIKU,
        self::CLAUDE_3_5_HAIKU      => 0.3,

        self::GEMINI_2_5_FLASH_PREVIEW_05_20,
        self::DEEPSEEK_CHAT         => 0.3,

        // Standart
        self::GPT_4_O,
        self::GPT_4_1,
        self::GPT_5,
        self::GPT_5_CHAT,
        self::GPT_5_1,
        self::GPT_5_1_CHAT          => 1.0,

        self::CLAUDE_3_5_SONNET,
        self::CLAUDE_3_5_SONNET_V2,
        self::CLAUDE_3_SONNET,
        self::CLAUDE_SONNET_4,
        self::CLAUDE_SONNET_4_5,
        self::CLAUDE_SONNET_4_6     => 1.0,

        self::GEMINI_2_5_PRO        => 1.0,

        // Premium (yuksek maliyet)
        self::GPT_4,
        self::GPT_4_TURBO,
        self::GPT_O_1,
        self::GPT_O_3,
        self::GPT_5_PRO,
        self::GPT_5_2,
        self::GPT_5_2_PRO,
        self::GPT_5_4               => 3.0,

        self::CLAUDE_3_OPUS,
        self::CLAUDE_OPUS_4,
        self::CLAUDE_OPUS_4_1,
        self::CLAUDE_OPUS_4_5,
        self::CLAUDE_OPUS_4_6       => 2.0,

        self::DEEPSEEK_REASONER     => 2.0,

        // Reasoning (cok yuksek maliyet)
        self::GPT_4_O1_PREVIEW,
        self::GPT_O_4_MINI,
        self::GPT_O_03_mini         => 4.0,

        // =============================================
        // GORSEL MODELLERI - Gorsel bazli (WithImagesInterface)
        // =============================================

        self::DALL_E_2              => 25.0,
        self::DALL_E_3              => 50.0,
        self::GPT_IMAGE_1           => 60.0,
        self::GPT_IMAGE_1_5         => 80.0,

        self::BLACK_FOREST_LABS_FLUX_1_SCHNELL => 15.0,

        self::STABLE_DIFFUSION_V_1_6   => 20.0,
        self::STABLE_DIFFUSION_XL_1024_V_1_0 => 30.0,
        self::SD_3, self::SD_3_TURBO   => 35.0,
        self::SD_3_MEDIUM              => 35.0,
        self::SD_3_LARGE               => 40.0,
        self::SD_3_LARGE_TURBO         => 45.0,
        self::SD_3_5_LARGE             => 45.0,
        self::SD_3_5_LARGE_TURBO       => 50.0,
        self::SD_3_5_MEDIUM            => 40.0,
        self::CORE                     => 20.0,
        self::ULTRA                    => 60.0,

        self::CLIPDROP                 => 30.0,
        self::NOVITA                   => 25.0,
        self::FREEPIK                  => 40.0,
        self::PEBBLELY                 => 25.0,

        // =============================================
        // VIDEO MODELLERI - Video/saniye bazli
        // =============================================

        self::IMAGE_TO_VIDEO        => 80.0,
        self::SORA_2                => 100.0,
        self::SORA_2_PRO            => 250.0,
        self::SYNTHESIA             => 300.0,
        self::HEYGEN                => 150.0,

        // =============================================
        // SES MODELLERI - TTS/STT
        // =============================================

        self::TTS_1                 => 5.0,
        self::TTS_1_HD              => 10.0,
        self::WHISPER_1             => 0.5,

        // =============================================
        // EMBEDDING & OZEL
        // =============================================

        self::TEXT_EMBEDDING_ADA_002,
        self::TEXT_EMBEDDING_3_SMALL,
        self::TEXT_EMBEDDING_3_LARGE => 0.01,

        self::PLAGIARISMCHECK       => 3.0,

        // Deep Research
        self::O3_DEEP_RESEARCH,
        self::O4_MINI_DEEP_RESEARCH,
        self::GEMINI_DEEP_RESEARCH  => 5.0,

        // =============================================
        // VARSAYILAN: unitPrice bazli otomatik hesaplama
        // =============================================
        // unitPrice $0.00001 referans alinarak oransal deger uretilir
        // Admin bunlari DB uzerinden override edebilir
        default => max(0.1, round($this->unitPrice() / 0.00001, 2)),
    };
}

// MEVCUT - DEGISMEZ
public function creditIndex(): float
{
    return 1.0; // separated sistem icin her zaman 1.0
}
```

#### Adim 2: HasCreditLimit::getCreditIndex() Guncellenir

```php
// app/Domains/Entity/Concerns/HasCreditLimit.php

public function getCreditIndex(): float
{
    // Shared credit kullanicilari icin model-bazli maliyet carpani kullan
    if ($this->isSharedCreditUser()) {
        // Oncelik: DB override > EntityEnum varsayilan
        $dbOverride = $this->getSharedCreditIndexFromDb();
        if ($dbOverride !== null) {
            return $dbOverride;
        }
        return $this->creditEnum()->sharedCreditIndex();
    }

    // Separated sistem: mevcut davranis (1.0)
    return $this->creditEnum()->creditIndex();
}

/**
 * DB'deki admin override degerini kontrol et (cache'li).
 * shared_credit_costs tablosunda bu model icin override varsa onu dondur.
 */
private function getSharedCreditIndexFromDb(): ?float
{
    $costs = Cache::remember('shared_credit_costs', 3600, function () {
        return SharedCreditCost::all()->pluck('base_cost', 'entity_key')->toArray();
    });

    $key = $this->creditEnum()->slug();

    return isset($costs[$key]) ? (float) $costs[$key] : null;
}
```

### 7.4 Bu Yaklasimin Avantajlari

**1. Sifir Yeni Hesaplama Kodu**

13 calculate trait'e (HasWords, HasImages, HasTextToVideo, vb.) **hic dokunmaya gerek yok**. Hepsi zaten `getCreditIndex()` cagiriyor. Sadece bu method'un dondurecegi degeri degistiriyoruz.

**2. Separated Sistem %100 Korunur**

`isSharedCreditUser()` false dondugunde `creditIndex()` (1.0) donmeye devam eder. Mevcut kod **tam olarak ayni** calisir.

**3. Mevcut Akis Otomatik Calisir**

```
calculate()
  -> miktar * getCreditIndex()    // shared modda sharedCreditIndex() degerini kullanir
  -> calculatedInputCredit set edilir
  -> decreaseCredit(calculatedInputCredit) cagirilir
  -> shared_credits'ten duser
```

Yeni bir hesaplama zinciri yazmaya gerek yok.

**4. CostCalculator Basitlesiyor**

`CostCalculator` artik `sharedCreditIndex()` degerini dogrudan kullanir. Ayri bir maliyet hesaplama katmanina gerek kalmaz:

```php
// ONCEKI PLAN (karmasik):
CostCalculator::calculate($entity, $featureType, $qty, $quality)
  -> DB'den base_cost al
  -> quality multiplier uygula
  -> qty ile carp

// YENI PLAN (basit):
// getCreditIndex() zaten dogru degeri donduruyor!
// calculate() trait'i otomatik olarak dogru maliyeti hesapliyor.
// CostCalculator sadece onizleme ve raporlama icin gerekli.
```

**5. Admin Override Hala Mumkun**

`shared_credit_costs` tablosu sadece admin'in EntityEnum varsayilanlarini degistirmek istediginde kullanilir. Tablo bossa, `sharedCreditIndex()` degerleri gecerlidir.

**Oncelik sirasi:**
```
1. DB override (shared_credit_costs tablosu) -> varsa bunu kullan
2. Plan-bazli override (plan.shared_credit_model_overrides) -> varsa bunu kullan
3. EntityEnum::sharedCreditIndex() varsayilan -> her zaman mevcut
```

### 7.5 Akis Karsilastirmasi

```
SEPARATED (mevcut, degismez):
  100 kelime GPT-4o
    -> calculate(): 100 * creditIndex(1.0) = 100
    -> decreaseCredit(100)
    -> entity_credits['open_ai']['gpt_4_o']['credit'] -= 100

SHARED (yeni):
  100 kelime GPT-4o
    -> calculate(): 100 * sharedCreditIndex(1.0) = 100
    -> decreaseCredit(100)
    -> users.shared_credits -= 100

  100 kelime Claude Opus
    -> calculate(): 100 * sharedCreditIndex(2.0) = 200
    -> decreaseCredit(200)
    -> users.shared_credits -= 200

  1 gorsel DALL-E 3
    -> calculate(): 1 * sharedCreditIndex(50.0) = 50
    -> decreaseCredit(50)
    -> users.shared_credits -= 50

  1 video Sora 2
    -> calculate(): 1 * sharedCreditIndex(100.0) = 100
    -> decreaseCredit(100)
    -> users.shared_credits -= 100

  100 kelime GPT-4o-mini (ucuz model)
    -> calculate(): 100 * sharedCreditIndex(0.3) = 30
    -> decreaseCredit(30)
    -> users.shared_credits -= 30
```

### 7.6 Kullaniciya Gosterim Ornekleri

Bir kullanici 10,000 shared kredi ile su islemleri yapabilir:

| Islem | Model | sharedCreditIndex | Miktar | Tuketim | Kalan |
|-------|-------|-------------------|--------|---------|-------|
| Chat | GPT-4o-mini | 0.3 | 1000 kelime | 300 | 9,700 |
| Chat | GPT-4o | 1.0 | 500 kelime | 500 | 9,200 |
| Gorsel | DALL-E 3 | 50.0 | 2 adet | 100 | 9,100 |
| Chat | Claude Opus | 2.0 | 200 kelime | 400 | 8,700 |
| Video | Sora 2 | 100.0 | 1 adet | 100 | 8,600 |
| TTS | TTS-1 HD | 10.0 | 3 adet | 30 | 8,570 |
| Chat | GPT-o1 (reasoning) | 3.0 | 100 kelime | 300 | 8,270 |

### 7.7 Varsayilan sharedCreditIndex Kategorileri

| Kategori | Index Araligi | Ornekler |
|----------|---------------|----------|
| Cok Ucuz | 0.01 - 0.1 | Embedding modelleri |
| Ucuz | 0.3 | GPT-4o-mini, Haiku, Flash, Deepseek |
| Standart | 1.0 | GPT-4o, Sonnet, Gemini Pro |
| Pahali | 2.0 - 3.0 | Opus, GPT-o1, GPT-o3, Reasoner |
| Cok Pahali | 4.0 - 5.0 | Deep Research, Reasoning |
| Gorsel (Ucuz) | 15.0 - 25.0 | Flux Schnell, DALL-E 2, Novita |
| Gorsel (Standart) | 30.0 - 50.0 | DALL-E 3, SD 3.5, Freepik |
| Gorsel (Premium) | 60.0 - 80.0 | GPT Image, Ultra |
| Video | 80.0 - 300.0 | Sora, HeyGen, Synthesia |
| Ses | 0.5 - 10.0 | Whisper, TTS-1, TTS-1 HD |

### 7.8 Faz 1'e Etkisi

Bu optimizasyon Faz 1'in kapsamini **basitlestirir**:

| Onceki Plan | Yeni Plan (creditIndex ile) |
|-------------|---------------------------|
| `SharedCreditService` + `CostCalculator` ayri servisler | `CostCalculator` sadece onizleme/raporlama icin, hesaplama `getCreditIndex()` uzerinden otomatik |
| `shared_credit_costs` tablosu **zorunlu** (tum modellerin maliyeti burada) | `shared_credit_costs` tablosu **opsiyonel** (sadece admin override icin). Varsayilanlar EntityEnum'da |
| Seeder ile tum modeller icin DB kaydi olustur | Seeder opsiyonel, `sharedCreditIndex()` zaten varsayilan degerleri tasiyor |
| `CostCalculator::calculate()` her dusme isleminde cagrilir | `getCreditIndex()` zaten dogru degeri donduruyor, ekstra hesaplama yok |

### 7.9 Dikkate Alinmasi Gerekenler

**1. Kalite Carpani (Quality Multiplier)**

`sharedCreditIndex()` temel carpani verir. Kalite carpani (HD, low quality, vb.) icin ek mekanizma gerekir. Bu, `calculateCredit()` asamasinda veya `CostCalculator` uzerinden uygulanabilir:

```php
// Ornek: DALL-E 3 HD gorsel
// sharedCreditIndex = 50.0
// HD quality multiplier = 2.0
// calculate(): 1 * 50.0 = 50 (temel)
// kalite carpani uygulandiginda: 50 * 2.0 = 100
```

Kalite carpani `sharedCreditIndex` icinde degil, ayri bir mekanizmada tutulur (plan override veya settings).

**2. Yeni Model Eklendiginde**

Yeni bir model EntityEnum'a eklendiginde, `sharedCreditIndex()` icin de deger eklenmesi gerekir. `default` case'i `unitPrice()` bazli otomatik deger urettigi icin unutulsa bile mantikli bir varsayilan kullanilir.

**3. Admin Guncelleme Akisi**

```
Admin model maliyetini degistirmek istiyor
  -> Admin paneldeki tabloda yeni deger girer
  -> shared_credit_costs tablosuna kayit yazilir (DB override)
  -> Cache invalidate edilir (Cache::forget('shared_credit_costs'))
  -> getCreditIndex() artik DB'deki degeri dondurur
  -> EntityEnum::sharedCreditIndex() atlanir (DB oncelikli)
```

---

## Bolum 8: Engine & Entity Domain Analizi - Ek Optimizasyon Firsatlari

### 8.1 Mevcut Altyapida Kesif Edilen Hazir Yapilar

Engine ve Entity domain'lerini detayli inceledikten sonra, shared credit sistemi icin **zaten var olan ama kullanilmayan** veya **dogrudan leverage edilebilecek** yapilar kesfettim:

#### Kesfedilen Yapi 1: `AITokenType` Enum (HAZIR)

**Dosya:** `app/Enums/AITokenType.php`

```php
enum AITokenType: string
{
    case WORD = 'word';
    case IMAGE = 'image';
    case CHARACTER = 'character';
    case MINUTE = 'minute';
    case SECOND = 'second';
    case IMAGE_TO_VIDEO = 'image_to_video';
    case TEXT_TO_SPEECH = 'text_to_speech';
    case SPEECH_TO_TEXT = 'speech_to_text';
    case TEXT_TO_VIDEO = 'text_to_video';
    case VIDEO_TO_VIDEO = 'video_to_video';
    case VISION = 'vision';
    case PLAGIARISM = 'plagiarism';
    case PRESENTATION = 'presentation';
}
```

**Neden onemli:** Shared credit transaction'larinda `feature_type` alanini string yerine **bu enum'u dogrudan kullanabiliriz**. Yeni bir enum yaratmaya gerek yok — 13 feature tipi zaten tanimli.

**Optimizasyon:**
```php
// ONCEKI PLAN: feature_type string olarak 'word', 'image' vb.
$table->string('feature_type');

// OPTIMIZE: AITokenType enum'u dogrudan kullan
$table->string('feature_type'); // DB'de string, PHP'de cast
// Model'de: 'feature_type' => AITokenType::class
```

#### Kesfedilen Yapi 2: `EntityEnum::tokenType()` (HAZIR)

**Dosya:** `app/Domains/Entity/Enums/EntityEnum.php` (~satir 1436)

```php
public function tokenType(): AITokenType
{
    $driverClass = $this->driverClass();

    return match (true) {
        class_interface_exists($driverClass, WithImagesInterface::class)       => AITokenType::IMAGE,
        class_interface_exists($driverClass, WithWordsInterface::class)        => AITokenType::WORD,
        class_interface_exists($driverClass, WithCharsInterface::class)        => AITokenType::CHARACTER,
        class_interface_exists($driverClass, WithMinuteInterface::class)       => AITokenType::MINUTE,
        class_interface_exists($driverClass, WithSecondInterface::class)       => AITokenType::SECOND,
        class_interface_exists($driverClass, WithImageToVideoInterface::class) => AITokenType::IMAGE_TO_VIDEO,
        class_interface_exists($driverClass, WithTextToSpeechInterface::class) => AITokenType::TEXT_TO_SPEECH,
        // ... (13 tip)
    };
}
```

**Neden onemli:** Her modelin feature tipini **otomatik** dondurur. `DALL_E_3->tokenType()` = `AITokenType::IMAGE`. `GPT_4_O->tokenType()` = `AITokenType::WORD`. Shared credit'te feature type'i manuel tanimlamaya gerek yok — `tokenType()` zaten bunu yapiyor.

**Optimizasyon:**
```php
// ONCEKI PLAN: DeductionContext'te feature_type string olarak geciriliyordu
new DeductionContext(entity: $entity, featureType: 'word', ...)

// OPTIMIZE: EntityEnum'dan otomatik al
new DeductionContext(entity: $entity, featureType: $entity->tokenType(), ...)
// Hatta DeductionContext icinde otomatik:
class DeductionContext {
    public function __construct(public EntityEnum $entity, ...) {
        $this->featureType = $entity->tokenType(); // otomatik!
    }
}
```

#### Kesfedilen Yapi 3: `EntityEnum::subLabel()` (HAZIR)

**Dosya:** `app/Domains/Entity/Enums/EntityEnum.php` (~satir 1415)

```php
public function subLabel(): string
{
    return match (true) {
        class_interface_exists($driverClass, WithImagesInterface::class)       => __('Images'),
        class_interface_exists($driverClass, WithWordsInterface::class)        => __('Words'),
        class_interface_exists($driverClass, WithTextToVideoInterface::class)  => __('Text to Video'),
        class_interface_exists($driverClass, WithTextToSpeechInterface::class) => __('Text to Speech'),
        // ... (13 tip)
    };
}
```

**Neden onemli:** Frontend'de "Bu islem ~50 kredi tuketecek (Gorsel)" gostermek icin tip labelini ceviri destekli olarak zaten hazir.

#### Kesfedilen Yapi 4: `EntityEnum::tooltipHowToCalc()` (UYARLANABILIR)

**Dosya:** `app/Domains/Entity/Enums/EntityEnum.php` (~satir 1457)

```php
public function tooltipHowToCalc(): string
{
    return match (true) {
        class_interface_exists($driverClass, WithImagesInterface::class) => __('1 Credit = 1 Image'),
        class_interface_exists($driverClass, WithWordsInterface::class)  => __('1 Credit = 1 Word'),
        class_interface_exists($driverClass, WithCharsInterface::class)  => __('1 Credit = 1 Character'),
        // ...
    };
}
```

**Optimizasyon:** Shared credit kullanicilari icin bu tooltip farkli olmali:

```php
// ONERILEN: Yeni method veya mevcut method'un adaptasyonu
public function sharedCreditTooltip(): string
{
    $index = $this->sharedCreditIndex();
    $type = $this->subLabel();
    return __(':cost Credit per :type', ['cost' => $index, 'type' => $type]);
    // Ornek: "50 Credit per Image", "1 Credit per Word"
}
```

#### Kesfedilen Yapi 5: `EntityStatItem` (UYARLANABILIR)

**Dosya:** `app/Domains/Entity/EntityStatItem.php`

```php
class EntityStatItem
{
    public function forUser(?User $user): self
    public function forPlan(?Plan $plan): self
    public function forTeam(?Team $team): self
    public function list(): Collection          // Tipe gore filtrelenmis entity listesi
    public function totalCredits(): float       // O tipteki toplam kredi
    public function checkIfThereUnlimited(): bool
    public function entityCount(): int
    public function filterByEngine(EngineEnum $enum): static
}
```

Kullanim: `EntityStats::word()->forUser($user)->totalCredits()` → Toplam kelime kredileri

**Optimizasyon:** Shared credit kullanicilari icin `totalCredits()` **anlamsiz** (cunku tek havuz var). Yeni bir `SharedCreditStats` sinifi veya `EntityStatItem`'a shared mod eklenmeli:

```php
// ONERILEN: EntityStatItem'a shared credit desteği
public function totalCredits(): float
{
    $user = $this->getUser();

    // Shared credit kullanicisi icin: tek havuzdan, bu tipe dusen tahmini kullanim
    if ($user?->isSharedCreditUser()) {
        return $user->shared_credits; // Tek bakiye
    }

    // Mevcut: entity bazli toplam
    return $this->list()->sum(/* ... */);
}
```

#### Kesfedilen Yapi 6: `BaseDriver::calculateCredit()` Zinciri (KRITIK)

**Dosya:** `app/Domains/Entity/BaseDriver.php` (satir 180-188)

```php
public function calculateCredit(): static
{
    if (! $this->guest) {
        $this->ensureUserProvided();
    }
    $this->calculatedInputCredit = $this->calculate();
    return $this;
}
```

Bu method `calculate()` cagirir (trait'lerdeki `HasWords::calculate()`, `HasImages::calculate()` vb.) ve sonucu `calculatedInputCredit`'e set eder.

**Kritik akis:**
```
driver->input($data)->calculateCredit()
                         |
                         v
                   $this->calculate()        // HasWords, HasImages, vb.
                         |
                         v
               quantity * getCreditIndex()   // <- BURASI shared credit icin calisir
                         |
                         v
            $this->calculatedInputCredit = sonuc
                         |
                         v
               driver->decreaseCredit()      // calculatedInputCredit kullanilir
```

**Bu zincir tamamen korunur** — `getCreditIndex()` dogru degeri dondurdugunde her sey otomatik calisir.

#### Kesfedilen Yapi 7: `EntityCollectionMixin::includeUnlisted()` (DIKKAT)

**Dosya:** `app/Domains/Entity/Mixins/EntityCollectionMixin.php`

```php
public function includeUnlisted(): Closure
{
    return function (bool $condition): Collection|static {
        return $this->when(
            ! $condition,
            fn ($entities) => $entities->filter(
                fn (EntityDriverInterface&WithCreditInterface $driver) =>
                    $driver->enum() === $driver->creditEnum()
            )
        );
    };
}
```

Bu, `creditBy()` uzerinden "listed" entity'leri filtreler. Su an `creditBy()` hep `$this` dondurdugu icin etkisiz ama gelecekte degisebilir. **Shared credit icin bu filtreleme onemli degil** cunku tum modeller tek havuzdan harciyor.

#### Kesfedilen Yapi 8: Engine `HasCache` Trait (KULLANILABILIR)

**Dosya:** `app/Domains/Engine/Concerns/HasCache.php`

```php
trait HasCache
{
    public int $cacheTtl = 300;

    // User-scoped cache
    public function userCache(string $key, Closure $callback): mixed
    {
        $key = "user:{$userId}:{$key}";
        return Cache::remember($key, $this->cacheTtl, $callback);
    }

    public function forget(string $key, bool $userScoped = false): void
}
```

**Optimizasyon:** `SharedCreditService`'te maliyet cache'leme icin bu pattern'i takip edebiliriz:

```php
// Model maliyetlerini cache'le (degismez veri)
Cache::remember('shared_credit_costs', 3600, fn() => SharedCreditCost::all());

// Kullanici bakiyesini ASLA cache'leme (tutarlilik icin)
```

### 8.2 Optimizasyon Ozet Tablosu

| Mevcut Yapi | Konum | Shared Credit Icin Kullanim | Etki |
|-------------|-------|---------------------------|------|
| `AITokenType` enum | `app/Enums/AITokenType.php` | Transaction'larda `feature_type` olarak | Yeni enum yaratma geregi ortadan kalkar |
| `EntityEnum::tokenType()` | EntityEnum ~L1436 | Her modelin feature tipini otomatik belirler | `DeductionContext`'te `featureType` otomatik |
| `EntityEnum::subLabel()` | EntityEnum ~L1415 | Frontend maliyet onizlemede tip etiketi | Ceviri destekli UI label'lari hazir |
| `EntityEnum::tooltipHowToCalc()` | EntityEnum ~L1457 | Shared credit birim maliyeti gosterimi | Uyarlanarak kullanilabilir |
| `EntityEnum::defaultCreditForDemo()` | EntityEnum ~L1497 | Demo kullanicilari icin shared credit miktari | Tip bazli varsayilan degerler hazir |
| `EntityStatItem` | `app/Domains/Entity/EntityStatItem.php` | Shared credit bakiye gosterimi | Adapte edilir veya yeni sinif olusturulur |
| `BaseDriver::calculateCredit()` | BaseDriver L180 | Kredi hesaplama zinciri | Degismez, `getCreditIndex()` uzerinden otomatik calisir |
| `Entity::planModels()` | Entity model L66 | Plan-bazli model erisim kontrolu | Shared credit'te de model erisimi icin kullanilir |
| Engine `HasCache` | Engine/Concerns/HasCache.php | Maliyet cache pattern'i | User-scoped cache pattern takip edilir |

### 8.3 Teknik Plana Etkisi: Sadelesetirilmis Yapilar

Bu kesfedilen yapilar sayesinde **orijinal plandaki bazi yapilar gereksizlesir:**

#### A. DeductionContext Basitlesiyor

```php
// ONCEKI:
class DeductionContext {
    public function __construct(
        public EntityEnum $entity,
        public string $featureType,    // <-- Manuel geciliyordu
        public float $quantity,
        public ?string $quality = 'standard',
        public ?array $metadata = null,
    ) {}
}

// OPTIMIZE:
class DeductionContext {
    public readonly AITokenType $featureType;

    public function __construct(
        public EntityEnum $entity,
        public float $quantity,
        public ?string $quality = 'standard',
        public ?array $metadata = null,
    ) {
        $this->featureType = $entity->tokenType(); // Otomatik!
    }
}
```

#### B. shared_credit_transactions Tablosu Basitlesiyor

```php
// feature_type artik AITokenType enum olarak cast edilir
protected $casts = [
    'feature_type' => AITokenType::class,
];
```

#### C. Maliyet Onizleme Komponenti Basitlesiyor

```blade
{{-- ONCEKI: Entity ve feature type ayri ayri geciriliyordu --}}
<x-shared-credit-cost-preview entity="dall_e_3" feature-type="image" ... />

{{-- OPTIMIZE: Sadece entity yeter, feature-type otomatik --}}
<x-shared-credit-cost-preview entity="dall_e_3" ... />
{{-- Component icinde: $entity->tokenType() ile tipi bilir --}}
{{-- $entity->subLabel() ile "Images" labelini alir --}}
{{-- $entity->sharedCreditIndex() ile maliyeti alir --}}
```

#### D. CostCalculator Ciddi Olarak Basitlesiyor

```php
// ONCEKI: Ayri feature_type parametresi, ayri hesaplama
CostCalculator::calculate($entity, $featureType, $qty, $quality)

// OPTIMIZE: Entity her seyi biliyor
public function calculate(EntityEnum $entity, float $qty, ?string $quality = null): float
{
    $baseCost = $this->getBaseCost($entity);   // sharedCreditIndex() veya DB override
    $multiplier = $quality ? $this->getQualityMultiplier($entity, $quality) : 1.0;
    return $baseCost * $multiplier * $qty;
}

private function getBaseCost(EntityEnum $entity): float
{
    // 1. DB override varsa (admin degistirmis)
    $override = $this->getDbOverride($entity);
    if ($override !== null) return $override;

    // 2. EntityEnum varsayilan
    return $entity->sharedCreditIndex();
}
```

#### E. Frontend Kredi Gosterimi Basitlesiyor

`credit-list.blade.php`'de shared credit kullanicisi icin:

```php
// Mevcut: entity tipine gore ayri ayri kredi sorgulaniyor
$wordEntities = EntityStats::word()->forUser($user);
$imageEntities = EntityStats::image()->forUser($user);
$wordCreditsCount = $wordEntities->totalCredits();
$imageCreditsCount = $imageEntities->totalCredits();

// Shared credit: Tek bakiye, tip bilgisi sadece maliyet icin gerekli
if ($user->isSharedCreditUser()) {
    $sharedBalance = $user->shared_credits;
    // Tek sayi goster, bit.
}
```

### 8.4 Kacinilmasi Gereken Tuzaklar

Domain analizinden ortaya cikan dikkat edilmesi gerekenler:

**1. `once()` Kullanimi**

Bircok method `once()` ile per-request caching yapiyor (ornegin `EntityStatItem::list()`, `HasCredit::getFreshCredits()`). Shared credit bakiyesinde `once()` **kullanilmamali** cunku bir request icinde birden fazla kredi dusme islem olabilir (ornegin batch islemler).

**2. `creditBy()` ve `includeUnlisted()`**

`creditBy()` su an hep `$this` donduruyor ama yapisi "bir modelin kredisini baska bir modelden say" imkani veriyor. Shared credit'te bu **irrelevant** cunku tek havuz var. Ancak `getFreshCredits()`'te `includeUnlisted(false)` filtrelemesi var — shared credit'te bu filtreleme atlanmali.

**3. Team + User Kredi Birlestirme**

`HasCreditLimit::getCredit()` metodu user ve team kredilerini birlestiriyor (L52-88). Shared credit'te team'in de `shared_credits` alani varsa, benzer birlestirme mantigi gerekir. Ama separated'daki gibi model-bazli degil, tek sayi toplanir:

```php
// Separated: user.gpt_4_o.credit + team.gpt_4_o.credit
// Shared:    user.shared_credits + team.shared_credits (tek sayi)
```

**4. `Plan::getCredit()` ve `isUnlimitedCredit()`**

Mevcut plan kontrollerinde `getCredit($engineKey, $entityKey)` kullaniliyor. Shared credit planinda bu method **cagrilmamali** — yerine `plan.shared_credits_amount` ve plan bazli unlimited kontrolu yapilmali.

**5. Octane Uyumlulugu**

`HasCreditLimit` icinde `config('octane.enabled')` kontrolu var — Octane modunda `once()` kullanilmiyor (stateful worker). Shared credit servisinde de ayni pattern takip edilmeli.

### 8.5 Sonuc: Ne Kadar Kod Tasarrufu Saglanir

| Alan | Orijinal Plan | Optimizasyon Sonrasi |
|------|---------------|---------------------|
| Feature type tanimlama | Yeni enum veya string sabitler | `AITokenType` enum zaten var |
| Feature type mapping | Her model icin mapping tablosu | `EntityEnum::tokenType()` otomatik |
| UI label'lari | Yeni ceviri dosyalari | `EntityEnum::subLabel()` + `tooltipHowToCalc()` hazir |
| CostCalculator parametreleri | 4 parametre (entity, type, qty, quality) | 3 parametre (entity, qty, quality) - type otomatik |
| DeductionContext | 5 parametre (entity, featureType, qty, quality, metadata) | 4 parametre - featureType otomatik |
| DB migration (feature_type) | String alan | String alan, PHP'de AITokenType cast |
| Frontend tip gosterimi | Yeni helper/component | Mevcut `subLabel()` |
| Demo kredit degerleri | Yeni tanimlar | `defaultCreditForDemo()` referans |

**Toplam etki:** ~%30 daha az yeni kod, mevcut altyapi maksimum kullanilir.

---

> **Not:** Bu dokuman codebase analizine dayali bir teknik plan taslagi olup,
> implementasyon surecinde guncellenmesi beklenmektedir.
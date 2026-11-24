# Faz 1: Backend Altyapisi [TAMAMLANDI]

> **Durum:** TAMAMLANDI
> **Bagimlilik:** Yok
> **Referans:** [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) Bolum 3, 7, 8

## Ozet

Temel veritabani, model, servis ve trait modifikasyonlari. Mevcut separated sistem %100 korunarak, yanina shared credit altyapisi eklendi.

## Mimari Karar: creditIndex Optimizasyonu

`creditIndex()` methodu tum 13 calculate trait tarafindan `quantity * getCreditIndex()` seklinde kullaniliyor. Shared credit kullanicilari icin `getCreditIndex()` methodu `sharedCreditIndex()` degerini donduruyor. Boylece hicbir calculate trait'e dokunmadan tum modeller icin farkli maliyet carpanlari uygulanir.

```
SEPARATED (degismez):
  100 kelime GPT-4o -> 100 * creditIndex(1.0) = 100 -> entity_credits'ten duser

SHARED (yeni):
  100 kelime GPT-4o -> 100 * sharedCreditIndex(1.0) = 100 -> shared_credits'ten duser
  100 kelime Claude Opus -> 100 * sharedCreditIndex(2.0) = 200 -> shared_credits'ten duser
  1 gorsel DALL-E 3 -> 1 * sharedCreditIndex(40.0) = 40 -> shared_credits'ten duser
```

### Oncelik Sirasi (getCreditIndex)
```
1. DB override (shared_credit_costs tablosu) -> varsa bunu kullan
2. EntityEnum::sharedCreditIndex() varsayilan -> her zaman mevcut
```

### Kesfedilen Hazir Yapilar
- `AITokenType` enum (`app/Enums/AITokenType.php`) -> feature_type icin hazir (13 tip tanimli)
- `EntityEnum::tokenType()` -> her modelin feature tipini otomatik dondurur
- `EntityEnum::subLabel()` -> ceviri destekli UI label'lari ("Words", "Images", vb.)
- `EntityEnum::tooltipHowToCalc()` -> birim aciklama ("1 Credit = 1 Word")
- `BaseDriver::calculateCredit()` zinciri -> `getCreditIndex()` uzerinden otomatik calisir

## Veritabani

### Yeni Tablolar

**shared_credit_transactions** - Audit trail
```sql
id, user_id(FK), entity_key, engine_key, feature_type, action_type,
amount(12,4), balance_after(12,4), unit_cost(10,4), quantity(10,4),
quality, metadata(json), description, timestamps
INDEX: [user_id, created_at], [user_id, action_type]
```

**shared_credit_costs** - Admin override (opsiyonel, bossa sharedCreditIndex kullanilir)
```sql
id, entity_key(unique), engine_key, feature_type,
base_cost(10,4), quality_high_multiplier(6,2), quality_low_multiplier(6,2),
is_active, timestamps
```

### Mevcut Tablolara Eklenen Kolonlar

**users:** `credit_system_type` (default 'separated'), `shared_credits` (decimal 12,4), `shared_credits_expires_at` (nullable timestamp)

**plans:** `credit_system_type`, `shared_credits_amount` (decimal 12,4), `shared_credit_model_overrides` (json), `shared_credit_feature_limits` (json)

**teams:** `credit_system_type`, `shared_credits` (decimal 12,4)

## Model Degisiklikleri

**User (`app/Models/User.php`):**
- Fillable: `credit_system_type`, `shared_credits`, `shared_credits_expires_at`
- Casts: `shared_credits` -> float, `shared_credits_expires_at` -> datetime
- `isSharedCreditUser(): bool`
- `sharedCreditTransactions(): HasMany`

**Plan (`app/Models/Plan.php`):**
- Fillable: `credit_system_type`, `shared_credits_amount`, `shared_credit_model_overrides`, `shared_credit_feature_limits`
- Casts: `shared_credits_amount` -> float, `shared_credit_model_overrides` -> array, `shared_credit_feature_limits` -> array
- `isSharedCreditPlan(): bool`

**Team (`app/Models/Team/Team.php`):**
- Fillable: `credit_system_type`, `shared_credits`
- Casts: `shared_credits` -> float
- `isSharedCreditTeam(): bool`

**SharedCreditTransaction (`app/Models/SharedCreditTransaction.php`):**
- Casts: `amount`, `balance_after`, `unit_cost`, `quantity` -> float, `metadata` -> array
- Relationship: `user()` -> BelongsTo

**SharedCreditCost (`app/Models/SharedCreditCost.php`):**
- `scopeActive()` scope
- 5 dakika cache'li sorgu

## SharedCreditService

**Dosya:** `app/Services/SharedCredit/SharedCreditService.php`

| Method | Aciklama |
|--------|----------|
| `isEnabled(): bool` | `shared_credit_system_enabled` setting kontrolu |
| `getBalance(User): float` | Kullanici bakiyesi |
| `hasBalance(User, float): bool` | Yeterli bakiye kontrolu |
| `deduct(User, EntityEnum, float, ?array): ?SharedCreditTransaction` | Atomik kredi dusme (lockForUpdate) |
| `topUp(User, float, string, ?EntityEnum, string): SharedCreditTransaction` | Kredi ekleme + log |
| `logTransaction(...): SharedCreditTransaction` | Transaction kaydı olusturma |
| `getHistory(User, ?string, int): Collection` | Islem gecmisi (filtrelenebilir) |
| `getCostOverride(string): ?SharedCreditCost` | DB override kontrolu (5dk cache) |

## HasCreditLimit Modifikasyonlari

**Dosya:** `app/Domains/Entity/Concerns/HasCreditLimit.php`

Her method'a `if ($this->isSharedCreditUser())` guard'i eklendi. Mevcut kodun hicbir satiri degismedi.

| Method | Shared Credit Davranisi |
|--------|------------------------|
| `isSharedCreditUser()` | YENI - setting + user type kontrolu |
| `getCreditIndex()` | DB override veya `sharedCreditIndex()` dondurur |
| `getCredit()` | `['credit' => shared_credits, 'isUnlimited' => false]` |
| `getIsUnlimitedCredit()` | false dondurur |
| `decreaseCredit()` | SharedCreditService::deduct() + UserUsageCredit log (uyumluluk) |
| `increaseCredit()` | SharedCreditService::topUp() |
| `setCredit()` | shared_credits'i set eder |
| `setAsUnlimited()` | no-op (shared credit unlimited desteklemez) |

## CreditUpdater Entegrasyonu

**Dosya:** `app/Services/PaymentGateways/Contracts/CreditUpdater.php`

| Method | Shared Credit Davranisi |
|--------|------------------------|
| `creditIncreaseSubscribePlan()` | Shared plan: `shared_credits_amount` ekle, `credit_system_type='shared'` yap, transaction log |
| `creditDecreaseCancelPlan()` | Shared plan: kredileri dus (soft cancel destekli), `credit_system_type='separated'` yap |

Tum payment gateway'ler (Stripe, PayPal, Iyzico, vb.) bu trait'i kullandigi icin shared credit desteği otomatik.

## Race Condition Korunmasi

```php
DB::transaction(function () use ($user, ...) {
    $freshUser = User::query()->lockForUpdate()->find($user->id);
    if ($freshUser->shared_credits < $amount) return null;
    $freshUser->shared_credits -= $amount;
    $freshUser->save();
    // Transaction log yaz
});
```

## sharedCreditIndex Kategorileri

| Kategori | Index | Ornekler |
|----------|-------|----------|
| Embedding | 0.01 | text-embedding-ada-002, voyage-2 |
| Ekonomik text | 0.3 | GPT-4o-mini, Haiku, Flash, Deepseek |
| STT | 0.5 | Whisper-1, Isolator |
| Standart text | 1.0 | GPT-4o, Sonnet, Gemini Pro, Grok |
| Premium text | 2.0 | Opus, GPT-4 Turbo, Claude 2.x |
| Reasoning | 4.0 | o1, o3, o4-mini, Deep Research |
| TTS | 5.0-10.0 | TTS-1, ElevenLabs, Speechify |
| Gorsel ucuz | 15.0 | DALL-E 2, Flux Schnell, Pebblely |
| Presentation | 20.0 | Gamma AI |
| Gorsel standart | 40.0 | DALL-E 3, Flux Pro, SD 3.x, Novita |
| Gorsel premium | 80.0 | GPT Image 1.5, Ultra, Kontext Max |
| Video | 150.0 | Sora, Kling, VEO, HeyGen, Synthesia |
| Default | `max(0.1, unitPrice/0.00001)` | Yeni eklenen modeller otomatik |

## Extension Uyumlulugu

Tum extension'lar (PhotoStudio, AiAvatar, FashionStudio, Chatbot, vb.) ayni `Entity::driver()->forUser()` pattern'ini kullaniyor. `HasCreditLimit` trait'i uzerinden entegrasyon otomatik olarak tum extension'lara yansir.

## Olusturulan Dosyalar

| Dosya | Aciklama |
|-------|----------|
| `database/migrations/2026_04_01_000001_add_shared_credit_columns.php` | users/plans/teams'e shared credit kolonlari |
| `database/migrations/2026_04_01_000002_create_shared_credit_transactions_table.php` | Transaction log tablosu |
| `database/migrations/2026_04_01_000003_create_shared_credit_costs_table.php` | Admin cost override tablosu |
| `app/Models/SharedCreditTransaction.php` | Transaction modeli (casts, user relation) |
| `app/Models/SharedCreditCost.php` | Cost override modeli (scopeActive, cache) |
| `app/Services/SharedCredit/SharedCreditService.php` | Core servis: deduct, topUp, getHistory, getCostOverride |
| `tests/Feature/SharedCreditTest.php` | 18 Pest test |

## Degistirilen Dosyalar

| Dosya | Degisiklik |
|-------|------------|
| `app/Domains/Entity/Concerns/HasCreditLimit.php` | isSharedCreditUser() + 7 method'a if-guard |
| `app/Domains/Entity/Enums/EntityEnum.php` | sharedCreditIndex() methodu (~200 model) |
| `app/Services/PaymentGateways/Contracts/CreditUpdater.php` | Shared plan subscribe/cancel handling |
| `app/Models/User.php` | +3 fillable, +2 casts, isSharedCreditUser(), sharedCreditTransactions() |
| `app/Models/Plan.php` | +4 fillable, +3 casts, isSharedCreditPlan() |
| `app/Models/Team/Team.php` | +2 fillable, +1 cast, isSharedCreditTeam() |
| `database/factories/UserFactory.php` | sharedCredit(float $balance = 1000.0) state |
| `database/factories/PlanFactory.php` | sharedCredit(float $amount = 1000.0) state |

## Test Sonuclari

- `php artisan test --filter=SharedCreditTest` -> 18/18 pass (41 assertions)
- `php artisan test` -> 1198/1198 pass (8160 assertions) - 0 regression
- `vendor/bin/pint --dirty` -> stil uyumlu

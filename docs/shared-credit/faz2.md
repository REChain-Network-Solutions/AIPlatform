# Faz 2: Admin Panel - Plan Yonetimi

> **Durum:** Bekliyor
> **Bagimlilik:** Faz 1
> **Referans:** [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) Bolum 3.5, 3.8

## Ozet

Admin, plan olustururken "Shared Credit" veya "Separated Credit" secebilmeli. Shared secildiginde model bazli kredi yerine tek bir `shared_credits_amount` girilmeli.

## Mevcut Plan Form Yapisi

Plan olusturma **Livewire component** uzerinden calisiyor (4 adimli wizard):

- **Step 1:** Temel bilgiler (isim, fiyat, frekans, tip)
- **Step 2:** Ozellikler/AI tools
- **Step 3:** Gelismis ayarlar
- **Step 4:** Model bazli kredi atama (ai_models JSON)

### Kritik Dosyalar

| Dosya | Aciklama |
|-------|----------|
| `app/Livewire/Admin/Finance/Plan/SubscriptionPlanCreate.php` | Subscription plan Livewire wizard. `submit()` metodu ile kayit (L153-177) |
| `app/Livewire/Admin/Finance/Plan/TokenPackPlanCreate.php` | Token pack Livewire wizard. `submit()` metodu ile kayit (L69-89) |
| `resources/views/default/panel/admin/finance/plan/includes/step-first.blade.php` | Step 1 blade view |
| `resources/views/livewire/assign-view-credits.blade.php` | Model bazli kredi atama formu (step 4) |
| `resources/views/default/panel/admin/finance/plan/index.blade.php` | Plan listesi tablosu |

> **Not:** `AdminController::paymentPlansSave()` legacy'dir. Guncel plan CRUD tamamen Livewire component'leri uzerinden yapilir.

## Adimlar

### 2.1: Livewire Component'lere Shared Credit Property'leri

**Dosya:** `app/Livewire/Admin/Finance/Plan/SubscriptionPlanCreate.php`

```php
// Yeni property'ler (mount icinde $plan'dan yukle)
public string $credit_system_type = 'separated';
public float $shared_credits_amount = 0;

// mount() icinde:
$this->credit_system_type = $this->plan->credit_system_type ?? 'separated';
$this->shared_credits_amount = $this->plan->shared_credits_amount ?? 0;

// rules() icinde validation ekle:
'credit_system_type' => 'required|in:separated,shared',
'shared_credits_amount' => 'required_if:credit_system_type,shared|numeric|min:0',

// submit() icinde kayit:
$this->plan->credit_system_type = $this->credit_system_type;
if ($this->credit_system_type === 'shared') {
    $this->plan->shared_credits_amount = $this->shared_credits_amount;
}
```

**Dosya:** `app/Livewire/Admin/Finance/Plan/TokenPackPlanCreate.php`

Ayni property/validation/save degisiklikleri.

### 2.2: Step 1'e Credit System Type Toggle

**Dosya:** `resources/views/default/panel/admin/finance/plan/includes/step-first.blade.php`

Step 1 form'undaki "Global Settings" veya "Pricing" section'ina eklenir:

```blade
{{-- Credit System Type --}}
<x-forms.input
    type="select"
    name="credit_system_type"
    wire:model="credit_system_type"
    :label="__('Credit System')"
>
    <option value="separated">{{ __('Separated (Model Based)') }}</option>
    <option value="shared">{{ __('Shared Credit Pool') }}</option>
</x-forms.input>
```

> **Not:** `step-first.blade.php` icerisinde mevcut extension hook'lari var (`@includeIf('chatbot::partials.plan-features')`, `@includeIf('multi-model::partials.plans_option')`, vb.). Yeni toggle bunlarin yakinina, "Pricing" bolumune eklenir.

### 2.3: Shared Credit Amount Alani (Step 4 veya Step 1)

`credit_system_type = 'shared'` oldugunda Step 4'teki model bazli kredi formu (`assign-view-credits`) gizlenir, yerine:

```blade
{{-- wire:model ile Livewire property'sine baglanir --}}
<div x-show="$wire.credit_system_type === 'shared'" x-transition>
    <x-forms.input
        type="number"
        name="shared_credits_amount"
        wire:model="shared_credits_amount"
        step="0.01"
        :label="__('Shared Credits Amount')"
        :placeholder="__('e.g. 10000')"
    />
    <p class="text-xs text-foreground/60 mt-1">
        {{ __('Total credits users receive. Used across all AI models with different cost rates.') }}
    </p>
</div>
```

### 2.4: assign-view-credits Gizleme

**Dosya:** `resources/views/livewire/assign-view-credits.blade.php`

Bu view model bazli kredi atama stepper'larini icerir (engine'ler, modeller, unlimited checkbox'lari, maliyet hesaplama). Shared credit plan icin gizlenmeli:

```blade
{{-- Mevcut assign-view-credits icerigi --}}
@if(($plan->credit_system_type ?? 'separated') !== 'shared')
    {{-- Mevcut model bazli kredi formu --}}
    {{-- ... mevcut kod aynen kalir ... --}}
@else
    <div class="p-4 text-center text-foreground/60">
        <p>{{ __('This plan uses shared credits. Model-based credit assignment is not needed.') }}</p>
        <p class="text-lg font-bold mt-2">
            {{ number_format($plan->shared_credits_amount ?? 0) }} {{ __('shared credits') }}
        </p>
    </div>
@endif
```

### 2.5: Plan Listesi'nde Badge Gosterimi

**Dosya:** `resources/views/default/panel/admin/finance/plan/index.blade.php`

Mevcut tablo kolonlari: Status, Type, Name, Price, Frequency, Updated At, Actions.

"Type" kolonunun yanina veya ayri bir kolona badge eklenir:

```blade
<td>
    @if($plan->isSharedCreditPlan())
        <x-badge class="bg-primary text-primary-foreground">{{ __('Shared') }}</x-badge>
        <span class="text-xs text-foreground/60 ms-1">
            {{ number_format($plan->shared_credits_amount) }} {{ __('cr') }}
        </span>
    @else
        <x-badge variant="secondary">{{ __('Separated') }}</x-badge>
    @endif
</td>
```

> **Not:** `<x-badge>` component'inin varligini dogrulamak gerekir. Yoksa `<span class="badge ...">` kullanilir.

## CreditUpdater Entegrasyonu (Faz 1'de Yapildi)

Plan subscribe/cancel sirasinda shared credit handling zaten Faz 1'de eklendi:

```php
// creditIncreaseSubscribePlan() - shared plan ise shared_credits ekle, credit_system_type='shared' yap
// creditDecreaseCancelPlan() - shared plan ise shared_credits'ten dusme, credit_system_type='separated' yap
```

## Ilgili Dosyalar

| Dosya | Islem | Detay |
|-------|-------|-------|
| `app/Livewire/Admin/Finance/Plan/SubscriptionPlanCreate.php` | MODIFY | +2 property, +2 rule, submit() guncelle |
| `app/Livewire/Admin/Finance/Plan/TokenPackPlanCreate.php` | MODIFY | Ayni degisiklikler |
| `resources/views/default/panel/admin/finance/plan/includes/step-first.blade.php` | MODIFY | credit_system_type toggle ekle |
| `resources/views/livewire/assign-view-credits.blade.php` | MODIFY | shared plan icin gizleme + ozet |
| `resources/views/default/panel/admin/finance/plan/index.blade.php` | MODIFY | Shared/Separated badge |

## Test Plani

```
- [x] Plan olusturma: credit_system_type = shared ile subscription plan olustur
- [x] Plan olusturma: credit_system_type = shared ile token pack plan olustur
- [x] Plan duzenleme: shared plan'i duzenle, shared_credits_amount degistir
- [x] Plan duzenleme: shared -> separated degistirince assign-view-credits gorungisi
- [x] Plan listesi: shared badge gorunuyor mu
- [x] Validation: shared secilip amount girilmezse hata
- [x] Plan atama: shared plan'a subscribe olunca user.credit_system_type = shared olur (Faz 1 testi)
- [x] assign-view-credits: shared plan icin model bazli form gizlenmis
```

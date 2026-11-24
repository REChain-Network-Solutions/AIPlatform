# Faz 4: Kullanici Dashboard Frontend

> **Durum:** Bekliyor
> **Bagimlilik:** Faz 2
> **Referans:** [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) Bolum 3.9, 8.2

## Ozet

Kullanicinin shared credit bakiyesini gormesi, kullanim gecmisi, harcama dagilimi ve generator sayfalarinda maliyet onizleme.

## Mevcut Frontend Yapisi

### Credit List Component
- **Dosya:** `resources/views/default/components/credit-list.blade.php`
- Model bazli progress bar'lar (Words, Images, vb.)
- Team destegi: `:team="$team"` prop'u
- Modal ile detayli kredi breakdown
- `fetch()` ile dinamik veri yukleme

### Abonelik Sayfasi
- **Dosya:** `resources/views/default/panel/user/finance/subscriptionPlans.blade.php`
- Plan overview karti (aktif plan, yenileme tarihi, team status)
- `<x-credit-list>` component'i ile kalan krediler
- Filtre: Monthly, Yearly, Pre-Paid, Lifetime
- Payment gateway secim modalleri
- **Route'lar:** `dashboard.user.payment.subscription`, `dashboard.user.payment.startSubscriptionProcess`, `dashboard.user.payment.startPrepaidPaymentProcess`

### Hazir Yapilar (v2 Bolum 8'den)
- `AITokenType` enum -> feature_type icin hazir (`app/Enums/AITokenType.php`)
- `EntityEnum::tokenType()` -> her modelin feature tipini otomatik dondurur
- `EntityEnum::subLabel()` -> ceviri destekli UI label'lari ("Words", "Images", vb.)
- `EntityEnum::tooltipHowToCalc()` -> birim aciklama ("1 Credit = 1 Word")
- `EntityStatItem` -> mevcut kredi istatistik sinifi, adapte edilebilir

## Adimlar

### 4.1: Credit List Component Guncelleme

**Dosya:** `resources/views/default/components/credit-list.blade.php`

Mevcut component model bazli progress bar'lar gosteriyor. Shared credit kullanicilari icin tek bakiye gosterimi:

```blade
@if(auth()->user()?->isSharedCreditUser())
    {{-- Shared Credit: Tek bakiye --}}
    <div class="shared-credit-balance p-3">
        <div class="flex items-center gap-2">
            <x-tabler-coins class="w-5 h-5 text-primary" />
            <span class="text-lg font-bold">
                {{ number_format(auth()->user()->shared_credits, 0) }}
            </span>
            <span class="text-foreground/60 text-sm">{{ __('credits') }}</span>
        </div>

        {{-- Progress bar (plan baslangic miktarina gore) --}}
        @php
            $plan = auth()->user()->activePlan();
            $planCredits = $plan?->shared_credits_amount ?? 0;
        @endphp
        @if($planCredits > 0)
        <div class="w-full bg-foreground/10 rounded-full h-2 mt-2">
            <div class="bg-primary h-2 rounded-full transition-all"
                 style="width: {{ min(100, (auth()->user()->shared_credits / $planCredits) * 100) }}%">
            </div>
        </div>
        <div class="text-xs text-foreground/40 mt-1">
            {{ __(':used of :total used', [
                'used' => number_format($planCredits - auth()->user()->shared_credits, 0),
                'total' => number_format($planCredits, 0)
            ]) }}
        </div>
        @endif
    </div>
@else
    {{-- Mevcut separated credit barlar --}}
    {{-- ... mevcut kod aynen kalir ... --}}
@endif
```

### 4.2: Kullanim Gecmisi Sayfasi

**Yeni route:** `dashboard.user.shared-credit.history`
**Yeni controller:** `app/Http/Controllers/User/SharedCreditController.php`

```php
class SharedCreditController extends Controller
{
    public function history(Request $request)
    {
        $user = auth()->user();

        $query = $user->sharedCreditTransactions()->latest();

        if ($request->filled('action_type') && $request->action_type !== 'all') {
            $query->where('action_type', $request->action_type);
        }

        $transactions = $query->paginate(25);

        return view('panel.user.shared-credit.history', compact('transactions'));
    }

    public function breakdown(Request $request)
    {
        $user = auth()->user();
        $period = (int) $request->input('period', 7);

        $breakdown = SharedCreditTransaction::where('user_id', $user->id)
            ->where('action_type', 'deduct')
            ->where('created_at', '>=', now()->subDays($period))
            ->selectRaw('feature_type, SUM(ABS(amount)) as total')
            ->groupBy('feature_type')
            ->get();

        return response()->json($breakdown);
    }
}
```

**View:** `resources/views/default/panel/user/shared-credit/history.blade.php`

```blade
<x-card>
    <h3>{{ __('Credit Usage History') }}</h3>

    {{-- Filtreler (Alpine.js) --}}
    <div class="flex gap-2 mb-4" x-data="{ filter: '{{ request('action_type', 'all') }}' }">
        @foreach(['all' => __('All'), 'deduct' => __('Usage'), 'top_up' => __('Top-ups'), 'subscription' => __('Subscriptions')] as $value => $label)
        <a :href="`?action_type=${filter}`"
           @click.prevent="filter = '{{ $value }}'; window.location.search = `?action_type={{ $value }}`"
           class="px-3 py-1 rounded-button text-sm [&.active]:bg-foreground/5"
           :class="{ 'active': filter === '{{ $value }}' }">
            {{ $label }}
        </a>
        @endforeach
    </div>

    {{-- Islem tablosu --}}
    <div class="overflow-x-auto">
        <table class="w-full text-sm">
            <thead>
                <tr>
                    <th class="text-start">{{ __('Date') }}</th>
                    <th class="text-start">{{ __('Model') }}</th>
                    <th class="text-start">{{ __('Type') }}</th>
                    <th class="text-end">{{ __('Amount') }}</th>
                    <th class="text-end">{{ __('Balance') }}</th>
                </tr>
            </thead>
            <tbody>
                @foreach($transactions as $tx)
                <tr>
                    <td>{{ $tx->created_at->diffForHumans() }}</td>
                    <td>
                        @if($tx->entity_key)
                            {{ \App\Domains\Entity\Enums\EntityEnum::tryFromSlug($tx->entity_key)?->label() ?? $tx->entity_key }}
                        @else
                            {{ $tx->description ?? '-' }}
                        @endif
                    </td>
                    <td>{{ $tx->feature_type ? __($tx->feature_type) : '-' }}</td>
                    <td class="text-end {{ $tx->amount < 0 ? 'text-red-500' : 'text-green-500' }}">
                        {{ ($tx->amount > 0 ? '+' : '') . number_format($tx->amount, 1) }}
                    </td>
                    <td class="text-end">{{ number_format($tx->balance_after, 1) }}</td>
                </tr>
                @endforeach
            </tbody>
        </table>
    </div>

    {{ $transactions->links() }}
</x-card>
```

### 4.3: Harcama Dagilimi Widget

Feature type bazli harcama breakdown (gecmis sayfasina veya dashboard'a eklenebilir):

```blade
<x-card x-data="usageBreakdown()" x-init="load()">
    <h4>{{ __('Usage Breakdown') }}</h4>

    <select x-model="period" @change="load()" class="text-sm border rounded-input p-1 mb-3">
        <option value="7">{{ __('Last 7 days') }}</option>
        <option value="30">{{ __('Last 30 days') }}</option>
    </select>

    <template x-for="item in breakdown" :key="item.feature_type">
        <div class="flex items-center justify-between py-2 border-b border-border last:border-0">
            <span x-text="item.feature_type" class="capitalize"></span>
            <span class="font-bold" x-text="parseFloat(item.total).toLocaleString() + ' credits'"></span>
        </div>
    </template>

    <x-empty-state
        icon="tabler-chart-bar-off"
        :title="__('No usage data')"
        :description="__('Start using AI features to see your breakdown.')"
        show="!loading && breakdown.length === 0"
        x-cloak
    />
</x-card>

@push('script')
<script>
document.addEventListener('alpine:init', () => {
    Alpine.data('usageBreakdown', () => ({
        period: 7,
        breakdown: [],
        loading: false,

        async load() {
            this.loading = true;
            try {
                const res = await fetch(`/dashboard/user/shared-credit/breakdown?period=${this.period}`);
                this.breakdown = await res.json();
            } finally {
                this.loading = false;
            }
        }
    }));
});
</script>
@endpush
```

### 4.4: Generator Sayfalarinda Maliyet Gosterimi

Yeni blade component: `<x-shared-credit-cost-preview>`

**Dosya:** `resources/views/default/components/shared-credit-cost-preview.blade.php`

```blade
@props(['entity'])

@if(auth()->user()?->isSharedCreditUser())
    @php
        $entityEnum = \App\Domains\Entity\Enums\EntityEnum::tryFromSlug($entity);
        if (!$entityEnum) return;
        $cost = $entityEnum->sharedCreditIndex();
        $type = $entityEnum->subLabel();
    @endphp
    <div class="inline-flex items-center gap-1 text-xs text-foreground/60">
        <x-tabler-coins class="w-3.5 h-3.5" />
        <span>{{ __(':cost credits per :type', ['cost' => $cost, 'type' => strtolower($type)]) }}</span>
    </div>
@endif
```

Kullanim:
```blade
<x-shared-credit-cost-preview entity="dall-e-3" />
{{-- Cikti: "40 credits per images" --}}

<x-shared-credit-cost-preview entity="gpt-4o" />
{{-- Cikti: "1 credits per words" --}}
```

### 4.5: Islem Sonrasi Bakiye Guncelleme

**Backend:** `HasCreditLimit::decreaseCredit()` shared credit dusme basarili oldugunda browser event dispatch eder.

**JS event pattern:** Alpine.js `window` event'i ile navbar guncelleme:

```javascript
// decreaseCredit sonrasinda (Livewire component veya AJAX response'da)
window.dispatchEvent(new CustomEvent('shared-credit-updated', {
    detail: { balance: newBalance }
}));
```

```blade
{{-- Navbar/header'daki bakiye (credit-list icerisinde) --}}
<span x-data="{ balance: {{ auth()->user()->shared_credits ?? 0 }} }"
      @shared-credit-updated.window="balance = $event.detail.balance">
    <span x-text="Math.floor(balance).toLocaleString()"></span>
    <span class="text-foreground/60 text-xs">{{ __('credits') }}</span>
</span>
```

### 4.6: Abonelik Sayfasi Guncelleme

**Dosya:** `resources/views/default/panel/user/finance/subscriptionPlans.blade.php`

Mevcut plan kartlari icinde shared credit plan'lar farkli gosterim:

```blade
@if($plan->isSharedCreditPlan())
    <div class="plan-card">
        <h3>{{ $plan->name }}</h3>
        <div class="text-3xl font-bold text-primary">
            {{ number_format($plan->shared_credits_amount) }}
        </div>
        <span class="text-foreground/60">{{ __('Shared Credits') }}</span>
        <p class="text-sm text-foreground/60 mt-2">
            {{ __('Use across all AI models with flexible pricing') }}
        </p>
    </div>
@else
    {{-- Mevcut separated plan karti --}}
@endif
```

### 4.7: Menu Kaydi

User tarafinda "Credit History" sayfasi icin menu kaydi (menus tablosuna veya admin panel uzerinden):

```php
[
    'title' => 'Credit History',
    'route' => 'dashboard.user.shared-credit.history',
    'icon' => 'tabler-history',
    'parent_id' => /* user menu ID */,
    'condition' => 'shared_credit_user', // sadece shared credit kullanicilarina goster
]
```

## Dikkat Edilecekler (v2 Bolum 8.4'ten)

1. **`once()` Kullanilmamali:** Shared credit bakiyesinde `once()` per-request caching kullanma. Bir request'te birden fazla kredi dusme olabilir (batch islemler).

2. **Chat/Writer Maliyet Gosterimi:** Yanit uzunlugu onceden bilinemiyor. "Cost calculated after response" notu goster. Generator sayfalarinda sadece birim maliyet ("1 credit per word") goster.

3. **Octane Uyumlulugu:** `config('octane.enabled')` kontrolu HasCreditLimit'te zaten var. Shared credit icin de ayni pattern.

4. **`creditBy()` ve `includeUnlisted()`:** Shared credit'te bu filtreleme irrelevant (tek havuz). `getFreshCredits()` icindeki `includeUnlisted(false)` atlanmali.

## Ilgili Dosyalar

| Dosya | Islem | Detay |
|-------|-------|-------|
| `resources/views/default/components/credit-list.blade.php` | MODIFY | Shared credit tek bakiye |
| `resources/views/default/components/shared-credit-cost-preview.blade.php` | CREATE | Birim maliyet onizleme |
| `resources/views/default/panel/user/shared-credit/history.blade.php` | CREATE | Kullanim gecmisi |
| `resources/views/default/panel/user/finance/subscriptionPlans.blade.php` | MODIFY | Shared plan karti |
| `app/Http/Controllers/User/SharedCreditController.php` | CREATE | history, breakdown |
| `routes/panel.php` | MODIFY | User route'lari |

## Test Plani

```
- [ ] Credit list: shared user'da tek bakiye gorunuyor mu
- [ ] Credit list: separated user'da mevcut barlar aynen calisir mi
- [ ] Credit list: team prop'u ile shared credit gosterimi
- [ ] Gecmis sayfasi: transaction'lar dogru listeleniyor mu
- [ ] Gecmis sayfasi: filtreleme (all, deduct, top_up, subscription)
- [ ] Gecmis sayfasi: sayfalama
- [ ] Harcama dagilimi: feature_type bazli dogru toplam
- [ ] Maliyet onizleme: component dogru birim maliyet gosteriyor mu
- [ ] Maliyet onizleme: separated user'da component gizli
- [ ] Plan karti: shared plan farkli gorunuyor mu
- [ ] Bakiye event'i: islem sonrasi navbar guncelleniyor mu
- [ ] Abonelik sayfasi: shared plan icin dogru gosterim
```

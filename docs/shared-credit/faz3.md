# Faz 3: Admin Panel - Maliyet ve Kullanici Yonetimi

> **Durum:** Bekliyor
> **Bagimlilik:** Faz 1 (Faz 2 ile paralel gelistirilebilir)
> **Referans:** [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) Bolum 3.2, 3.8

## Ozet

SharedCreditCost admin CRUD, global ayarlar, kullanici bakiye yonetimi ve transaction log sayfasi.

## Mevcut Admin Yapi Bilgisi

### Route Yapisi
Admin route'lari `routes/panel.php` icerisinde, `dashboard.admin.` namespace'i altinda tanimli. Yeni route'lar ayni pattern'e uygun eklenir.

### Menu Sistemi
Navigation **database-driven** (`menus` tablosu + `MenuService`). Yeni sayfalar icin:
1. Migration ile `menus` tablosuna kayit eklenir, VEYA
2. Admin panelden Settings > Menu Manager ile eklenir

**MenuService:** `app/Services/Common/MenuService.php` - `generate()` ve `regenerate()` methodlari

### Admin Finance Config
- **Controller:** `app/Http/Controllers/Admin/Config/FinanceController.php`
- **Method:** `store(Request $request)` (L26-60)
- **View:** `resources/views/default/panel/admin/config/finance.blade.php`
- **Route:** `dashboard.admin.config.finance.store` (POST)
- Mevcut section'lar: Affiliate Settings, Billing, Plans Settings

### Admin User Finance
- **View:** `resources/views/default/panel/admin/users/finance.blade.php`
- 3 section: User Information (readonly), Manage Subscription, Token Pack
- **Route'lar:** `dashboard.user.payment.assignPlanByAdmin`, `dashboard.user.payment.cancelActiveSubscriptionByAdmin`, `dashboard.user.payment.assignTokenByAdmin`

## Adimlar

### 3.1: SharedCreditCost Admin CRUD

Admin'in model bazli maliyet carpanlarini override etmesi icin.

**Yeni Controller:** `app/Http/Controllers/Admin/Finance/SharedCreditCostController.php`

```php
class SharedCreditCostController extends Controller
{
    public function index()       // Tum modeller + mevcut override'lar listesi
    public function store()       // Yeni override olustur
    public function update()      // Override guncelle
    public function destroy()     // Override sil (sharedCreditIndex varsayilana doner)
}
```

**Yeni Route'lar:** `routes/panel.php` (admin grubu icinde)

```php
Route::prefix('admin/finance/shared-credit-costs')
    ->name('dashboard.admin.finance.shared-credit-costs.')
    ->group(function () {
        Route::get('/', [SharedCreditCostController::class, 'index'])->name('index');
        Route::post('/', [SharedCreditCostController::class, 'store'])->name('store');
        Route::put('/{cost}', [SharedCreditCostController::class, 'update'])->name('update');
        Route::delete('/{cost}', [SharedCreditCostController::class, 'destroy'])->name('destroy');
    });
```

**View:** `resources/views/default/panel/admin/finance/shared-credit-costs/index.blade.php`

Tablo yapisi - `EntityEnum::cases()` uzerinden tum modeller listelenir:

```
| Model | Engine | Feature Type | Varsayilan (sharedCreditIndex) | DB Override | Aktif | Islem |
|-------|--------|-------------|-------------------------------|-------------|-------|-------|
| GPT-4o | OpenAI | Word | 1.0 | - | - | [Edit] |
| GPT-4o-mini | OpenAI | Word | 0.3 | 0.5 | Y | [Edit] [Sil] |
| DALL-E 3 | OpenAI | Image | 40.0 | 35.0 | Y | [Edit] [Sil] |
```

**Hazir yapilar kullanilir:**
- `EntityEnum::tokenType()` -> feature_type otomatik (`AITokenType` enum)
- `EntityEnum::subLabel()` -> UI label'lari ("Words", "Images", vb.)
- `EntityEnum::engine()` -> engine bilgisi otomatik
- `EntityEnum::label()` -> model gorunum adi

**Cache invalidation:** Override olusturulunca/guncellennce/silinince:
```php
Cache::forget("shared_credit_cost:{$entityKey}");
```

### 3.2: Admin Settings - Global Toggle

**Dosya:** `resources/views/default/panel/admin/config/finance.blade.php`

Mevcut section'larin altina (Affiliate, Billing, Plans Settings) yeni "Shared Credit System" section'i:

```blade
<x-card>
    <h4>{{ __('Shared Credit System') }}</h4>

    <x-forms.input
        type="checkbox"
        name="shared_credit_system_enabled"
        :checked="setting('shared_credit_system_enabled')"
        :label="__('Enable Shared Credit System')"
        switcher
    />

    <x-forms.input
        type="checkbox"
        name="shared_credit_show_cost_preview"
        :checked="setting('shared_credit_show_cost_preview', true)"
        :label="__('Show cost preview before operations')"
        switcher
    />

    <x-forms.input
        type="checkbox"
        name="shared_credit_allow_negative"
        :checked="setting('shared_credit_allow_negative')"
        :label="__('Allow negative balance')"
        switcher
    />

    <x-forms.input
        type="number"
        name="shared_credit_expiration_days"
        :value="setting('shared_credit_expiration_days', 0)"
        :label="__('Credit expiration (days, 0 = never)')"
    />
</x-card>
```

**Controller:** `app/Http/Controllers/Admin/Config/FinanceController.php` -> `store()` metoduna yeni alanlar:

```php
// store() icinde mevcut setting kayitlarinin yanina:
setting_save('shared_credit_system_enabled', $request->has('shared_credit_system_enabled'));
setting_save('shared_credit_show_cost_preview', $request->has('shared_credit_show_cost_preview'));
setting_save('shared_credit_allow_negative', $request->has('shared_credit_allow_negative'));
setting_save('shared_credit_expiration_days', (int) $request->input('shared_credit_expiration_days', 0));
```

### 3.3: Admin Kullanici Finance Sayfasi

**Dosya:** `resources/views/default/panel/admin/users/finance.blade.php`

Mevcut 3 section'in altina yeni "Shared Credit" section'i eklenir:

```blade
@if($user->isSharedCreditUser() || setting('shared_credit_system_enabled'))
<x-card class="mt-4">
    <h4>{{ __('Shared Credit Management') }}</h4>

    {{-- Bakiye gosterimi --}}
    <div class="flex items-center gap-4 mb-4">
        <div>
            <span class="text-2xl font-bold">{{ number_format($user->shared_credits, 2) }}</span>
            <span class="text-foreground/60">{{ __('credits') }}</span>
        </div>
        <x-badge :class="$user->isSharedCreditUser() ? 'bg-primary text-primary-foreground' : 'bg-secondary'">
            {{ $user->credit_system_type }}
        </x-badge>
    </div>

    {{-- Manuel bakiye ayarlama --}}
    <form method="POST" action="{{ route('dashboard.admin.users.shared-credit.adjust', $user) }}">
        @csrf
        <div class="grid grid-cols-3 gap-4">
            <x-forms.input type="number" name="amount" step="0.01" :label="__('Amount')" required />
            <x-forms.input type="select" name="action" :label="__('Action')">
                <option value="add">{{ __('Add Credits') }}</option>
                <option value="subtract">{{ __('Subtract Credits') }}</option>
                <option value="set">{{ __('Set Balance To') }}</option>
            </x-forms.input>
            <x-forms.input type="text" name="description" :label="__('Reason')" />
        </div>
        <x-button type="submit" class="mt-2">{{ __('Apply') }}</x-button>
    </form>

    {{-- Credit system type degistirme --}}
    <form method="POST" action="{{ route('dashboard.admin.users.credit-type.update', $user) }}" class="mt-4">
        @csrf @method('PUT')
        <div class="flex items-center gap-4">
            <x-forms.input type="select" name="credit_system_type" :label="__('Credit System')">
                <option value="separated" @selected(!$user->isSharedCreditUser())>{{ __('Separated') }}</option>
                <option value="shared" @selected($user->isSharedCreditUser())>{{ __('Shared') }}</option>
            </x-forms.input>
            <x-button type="submit">{{ __('Update Type') }}</x-button>
        </div>
    </form>

    {{-- Son 10 islem --}}
    @if($user->isSharedCreditUser())
    <h5 class="mt-4">{{ __('Recent Transactions') }}</h5>
    <div class="overflow-x-auto">
        <table class="w-full text-sm">
            <thead>
                <tr>
                    <th>{{ __('Date') }}</th>
                    <th>{{ __('Model') }}</th>
                    <th>{{ __('Action') }}</th>
                    <th>{{ __('Amount') }}</th>
                    <th>{{ __('Balance') }}</th>
                </tr>
            </thead>
            <tbody>
                @foreach($user->sharedCreditTransactions()->latest()->limit(10)->get() as $tx)
                <tr>
                    <td>{{ $tx->created_at->format('Y-m-d H:i') }}</td>
                    <td>{{ $tx->entity_key ?? '-' }}</td>
                    <td>{{ $tx->action_type }}</td>
                    <td class="{{ $tx->amount < 0 ? 'text-red-500' : 'text-green-500' }}">
                        {{ ($tx->amount > 0 ? '+' : '') . number_format($tx->amount, 2) }}
                    </td>
                    <td>{{ number_format($tx->balance_after, 2) }}</td>
                </tr>
                @endforeach
            </tbody>
        </table>
    </div>
    @endif
</x-card>
@endif
```

**Yeni route'lar:**
```php
Route::post('admin/users/{user}/shared-credit/adjust', [UserSharedCreditController::class, 'adjust'])
    ->name('dashboard.admin.users.shared-credit.adjust');
Route::put('admin/users/{user}/credit-type', [UserSharedCreditController::class, 'updateType'])
    ->name('dashboard.admin.users.credit-type.update');
```

**Yeni controller:** `app/Http/Controllers/Admin/UserSharedCreditController.php`

```php
class UserSharedCreditController extends Controller
{
    public function adjust(Request $request, User $user): RedirectResponse
    {
        $validated = $request->validate([
            'amount' => 'required|numeric|min:0.01',
            'action' => 'required|in:add,subtract,set',
            'description' => 'nullable|string|max:255',
        ]);

        $service = app(SharedCreditService::class);
        match ($validated['action']) {
            'add' => $service->topUp($user, $validated['amount'], $validated['description'] ?? 'Admin adjustment', null, 'admin_adjust'),
            'subtract' => $service->deduct($user, /* ... */),
            'set' => /* set credit logic */,
        };

        return back()->with('message', __('Credits updated.'));
    }

    public function updateType(Request $request, User $user): RedirectResponse
    {
        $validated = $request->validate([
            'credit_system_type' => 'required|in:separated,shared',
        ]);

        $user->credit_system_type = $validated['credit_system_type'];
        $user->save();

        return back()->with('message', __('Credit system type updated.'));
    }
}
```

### 3.4: Transaction Log Sayfasi

**Yeni route:** `dashboard.admin.finance.shared-credit-transactions.index`

**Yeni controller:** `app/Http/Controllers/Admin/Finance/SharedCreditTransactionController.php`

```php
class SharedCreditTransactionController extends Controller
{
    public function index(Request $request)
    {
        $query = SharedCreditTransaction::query()->with('user:id,name,email');

        // Filtreler
        if ($request->filled('user')) {
            $query->whereHas('user', fn($q) => $q->where('name', 'like', "%{$request->user}%")
                ->orWhere('email', 'like', "%{$request->user}%"));
        }
        if ($request->filled('action_type')) {
            $query->where('action_type', $request->action_type);
        }
        if ($request->filled('from')) {
            $query->where('created_at', '>=', $request->from);
        }
        if ($request->filled('to')) {
            $query->where('created_at', '<=', $request->to);
        }

        $transactions = $query->latest()->paginate(25);

        // Istatistikler
        $stats = [
            'total_deducted_today' => SharedCreditTransaction::where('action_type', 'deduct')
                ->whereDate('created_at', today())->sum(DB::raw('ABS(amount)')),
            'total_topped_up_today' => SharedCreditTransaction::where('action_type', 'top_up')
                ->whereDate('created_at', today())->sum('amount'),
        ];

        return view('panel.admin.finance.shared-credit-transactions.index', compact('transactions', 'stats'));
    }
}
```

**View:** `resources/views/default/panel/admin/finance/shared-credit-transactions/index.blade.php`

- Filtre bar: user arama, tarih araligi, action_type secimi
- Istatistik kartlari: Bugunun harcamasi, top-up, en aktif modeller
- Sayfalamali tablo: Date, User, Model, Action, Amount, Balance After
- Opsiyonel: CSV export

### 3.5: Menu Kayitlari

Database-driven menu sistemi icin migration veya seeder:

```php
// Migration veya seeder ile menus tablosuna kayitlar
// parent: Admin > Finance
[
    'title' => 'Shared Credit Costs',
    'route' => 'dashboard.admin.finance.shared-credit-costs.index',
    'icon' => 'tabler-coins',
    'parent_id' => /* Finance menu item ID */,
    'role' => 'admin',
],
[
    'title' => 'Shared Credit Transactions',
    'route' => 'dashboard.admin.finance.shared-credit-transactions.index',
    'icon' => 'tabler-receipt',
    'parent_id' => /* Finance menu item ID */,
    'role' => 'admin',
],
```

Alternatif: Admin panelden Settings > Menu Manager ile el ile eklenir.

## Ilgili Dosyalar

| Dosya | Islem | Detay |
|-------|-------|-------|
| `app/Http/Controllers/Admin/Finance/SharedCreditCostController.php` | CREATE | Model maliyet CRUD |
| `app/Http/Controllers/Admin/Finance/SharedCreditTransactionController.php` | CREATE | Transaction log sayfasi |
| `app/Http/Controllers/Admin/UserSharedCreditController.php` | CREATE | User bakiye yonetimi |
| `resources/views/default/panel/admin/finance/shared-credit-costs/index.blade.php` | CREATE | Maliyet tablosu |
| `resources/views/default/panel/admin/finance/shared-credit-transactions/index.blade.php` | CREATE | Transaction log |
| `resources/views/default/panel/admin/config/finance.blade.php` | MODIFY | Global settings toggle'lari |
| `app/Http/Controllers/Admin/Config/FinanceController.php` | MODIFY | store() -> yeni setting kayitlari |
| `resources/views/default/panel/admin/users/finance.blade.php` | MODIFY | Shared credit section |
| `routes/panel.php` | MODIFY | Yeni admin route'lari |
| `database/migrations/` veya `database/seeders/` | CREATE | Menu kayitlari (opsiyonel) |

## Test Plani

```
- [ ] SharedCreditCost CRUD: override olustur, guncelle, sil
- [ ] SharedCreditCost: silince sharedCreditIndex varsayilana doner
- [ ] Cache invalidation: override degisince cache temizlenir
- [ ] Settings: shared_credit_system_enabled toggle acik/kapali
- [ ] Settings: tum shared credit setting'leri kaydediliyor
- [ ] User finance: bakiye gosterim dogru
- [ ] User finance: manuel kredi ekle/cikar/set
- [ ] User finance: credit system type degistirme (shared <-> separated)
- [ ] Transaction log: filtreleme (user, tarih, action_type)
- [ ] Transaction log: sayfalama
- [ ] Transaction log: istatistikler dogru hesaplaniyor
```

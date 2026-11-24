# Shared Credit Migration Plan

Mevcut kullanicilari ve planlari **separated** kredi sisteminden **shared** kredi sistemine gecirme plani.

Admin panelinde **Livewire wizard form** ile adim adim migration.

---

## Kararlar

| Soru | Karar |
|------|-------|
| Migration arayuzu | Admin panelinde wizard form (Livewire `WithSteps` trait) |
| Migration ilerleme dashboard'u | Evet - wizard icinde gosterilir |
| Kullanici bildirimi | Opsiyonel - newsletter uzerinden, wizard son adiminda secilir |
| Downgrade (shared → separated) | Desteklenecek (rollback butonu) |
| Donusum formulu | Global weighted sum (`entity_credit x sharedCreditIndex`) |
| Unlimited modeller | Wizard'da listelenir, admin her model icin kredi miktarini girer |
| Free plan kullanicilari | Evet, onlar da migrate edilecek |

---

## Route & Erisim

```
URL:   /dashboard/admin/finance/shared-credit-migration
Route: dashboard.admin.finance.shared-credit-migration
```

Admin Finance menusune "Credit Migration" linki eklenir.

---

## Wizard Adimlari

```
[1. Analiz] → [2. Plan Yapilandirma] → [3. Kullanici Gecisi] → [4. Onizleme & Calistir] → [5. Bildirim]
   25%              50%                       75%                      90%                     100%
```

---

## Adim 1: Analiz

**Amac:** Mevcut durumu goster, neyin migrate edilecegini ozetle.

### Gorunum

```
+-----------------------------------------------------------+
| Step 1: Analysis                                          |
+-----------------------------------------------------------+
|                                                           |
| Current System Status                                     |
| ┌───────────────────────────────────────────────────────┐ |
| │ Total Plans:          12                              │ |
| │ Separated Plans:      8  (migrate edilecek)           │ |
| │ Already Shared:       4  (zaten shared)               │ |
| │                                                       │ |
| │ Total Active Users:   230                             │ |
| │ Separated Users:      174 (migrate edilecek)          │ |
| │ Already Shared:       56  (zaten shared)              │ |
| │                                                       │ |
| │ Team Plans:           3                               │ |
| │ Free Plans:           2                               │ |
| └───────────────────────────────────────────────────────┘ |
|                                                           |
| Plans to Migrate                                          |
| ┌──────┬──────────────┬────────┬───────────┬───────────┐ |
| │  ☑   │ Plan Name    │ Price  │ Users     │ Unlimited │ |
| ├──────┼──────────────┼────────┼───────────┼───────────┤ |
| │  ☑   │ Basic Monthly│ $9.99  │ 45 users  │ -         │ |
| │  ☑   │ Pro Monthly  │ $29.99 │ 82 users  │ 2 models  │ |
| │  ☑   │ Enterprise   │ $99.99 │ 30 users  │ 5 models  │ |
| │  ☑   │ Free Trial   │ $0     │ 15 users  │ -         │ |
| │  ☐   │ Legacy Plan  │ $4.99  │ 2 users   │ -         │ |
| └──────┴──────────────┴────────┴───────────┴───────────┘ |
|                                                           |
| ⓘ Select the plans you want to migrate.                  |
| ⓘ Plans already on shared credit are not shown.          |
|                                                           |
|                              [Back]  [Next: Configure →]  |
+-----------------------------------------------------------+
```

### Livewire Data

```php
// Mount'da hesaplanir
public array $plans = [];          // Separated planlar
public array $selectedPlanIds = []; // Secili plan ID'leri
public int $totalSeparatedUsers = 0;
public int $totalSharedUsers = 0;

public function mount(): void
{
    $this->plans = Plan::where('credit_system_type', 'separated')
        ->withCount(['subscriptions as active_users_count' => fn($q) =>
            $q->whereIn('stripe_status', ['active', 'trialing', 'free_approved', ...])
        ])
        ->get()
        ->map(fn($plan) => [
            'id' => $plan->id,
            'name' => $plan->name,
            'price' => $plan->price,
            'frequency' => $plan->frequency,
            'active_users' => $plan->active_users_count,
            'unlimited_models' => $this->getUnlimitedModels($plan),
            'suggested_amount' => $this->calculateSuggestedAmount($plan),
            'selected' => true,
        ])->toArray();

    $this->selectedPlanIds = collect($this->plans)->pluck('id')->toArray();
}
```

---

## Adim 2: Plan Yapilandirma

**Amac:** Her secili plan icin shared credit miktarini belirle.

### Gorunum

```
+-----------------------------------------------------------+
| Step 2: Plan Configuration                                |
+-----------------------------------------------------------+
|                                                           |
| Configure shared credit amounts for each plan.            |
|                                                           |
| ┌─────────────────────────────────────────────────────┐   |
| │ Basic Monthly ($9.99/mo)                            │   |
| │                                                     │   |
| │ Current entity credits:                             │   |
| │   GPT-4o: 500 × 1.0  = 500.0                      │   |
| │   GPT-4o Mini: 1000 × 0.3  = 300.0                │   |
| │   DALL-E 3: 50 × 40.0  = 2,000.0                  │   |
| │   ─────────────────────────────────                 │   |
| │   Suggested total: 2,800.0 credits                  │   |
| │                                                     │   |
| │ Shared Credits Amount: [________2800_________]      │   |
| │                                                     │   |
| │ ☑ Reset credits on renewal                          │   |
| └─────────────────────────────────────────────────────┘   |
|                                                           |
| ┌─────────────────────────────────────────────────────┐   |
| │ Pro Monthly ($29.99/mo)                             │   |
| │                                                     │   |
| │ Current entity credits:                             │   |
| │   GPT-4o: 2000 × 1.0  = 2,000.0                   │   |
| │   Claude Sonnet: 1000 × 1.0  = 1,000.0            │   |
| │   DALL-E 3: 200 × 40.0  = 8,000.0                 │   |
| │   ─────────────────────────────────                 │   |
| │   Suggested total: 11,000.0 credits                 │   |
| │                                                     │   |
| │ ⚠ Unlimited Models (assign credits manually):      │   |
| │   GPT-5:        [________5000_________]             │   |
| │   Claude Opus:  [________3000_________]             │   |
| │                                                     │   |
| │ Shared Credits Amount: [________19000________]      │   |
| │ (11,000 + 5,000 + 3,000)                           │   |
| │                                                     │   |
| │ ☑ Reset credits on renewal                          │   |
| └─────────────────────────────────────────────────────┘   |
|                                                           |
|                         [← Back]  [Next: Users →]         |
+-----------------------------------------------------------+
```

### Livewire Data

```php
public array $planConfigs = [];
// Her plan icin:
// [
//   'plan_id' => 1,
//   'suggested_amount' => 2800.0,
//   'shared_credits_amount' => 2800.0,  // admin duzenleyebilir
//   'unlimited_models' => [
//       ['slug' => 'gpt-5', 'label' => 'GPT-5', 'amount' => 0],
//   ],
//   'reset_credits_on_renewal' => true,
//   'breakdown' => [
//       ['entity' => 'GPT-4o', 'credit' => 500, 'index' => 1.0, 'value' => 500.0],
//       ...
//   ],
// ]
```

### Hesaplama Formulu

```php
private function calculateSuggestedAmount(Plan $plan): float
{
    $total = 0;
    foreach ($plan->ai_models as $engineModels) {
        foreach ($engineModels as $entitySlug => $config) {
            if ($config['isUnlimited'] ?? false) {
                continue; // unlimited modeller ayri gosterilir
            }
            try {
                $entity = EntityEnum::fromSlug($entitySlug);
                $credit = (float) ($config['credit'] ?? 0);
                $total += $credit * $entity->sharedCreditIndex();
            } catch (\Throwable) {
                continue; // bilinmeyen entity, atla
            }
        }
    }
    return round($total, 2);
}

private function getUnlimitedModels(Plan $plan): array
{
    $unlimited = [];
    foreach ($plan->ai_models as $engineModels) {
        foreach ($engineModels as $entitySlug => $config) {
            if ($config['isUnlimited'] ?? false) {
                $entity = EntityEnum::fromSlug($entitySlug);
                $unlimited[] = [
                    'slug' => $entitySlug,
                    'label' => $entity->label(),
                    'amount' => 0,
                ];
            }
        }
    }
    return $unlimited;
}
```

---

## Adim 3: Kullanici Gecisi

**Amac:** Kullanicilarin nasil gecirilecegini sec.

### Gorunum

```
+-----------------------------------------------------------+
| Step 3: User Migration                                    |
+-----------------------------------------------------------+
|                                                           |
| How should active users be migrated?                      |
|                                                           |
| ◉ Migrate immediately (pro-rata)                         |
|   Users get shared credits based on remaining cycle.      |
|   Example: 15/30 days left = 50% of plan amount.         |
|                                                           |
| ○ Wait for next renewal                                   |
|   Users continue with separated credits until their       |
|   subscription renews. CreditUpdater handles the switch.  |
|                                                           |
| ─────────────────────────────────────────────────────     |
|                                                           |
| Summary per plan:                                         |
| ┌──────────────────┬────────┬──────────┬─────────────┐   |
| │ Plan             │ Users  │ Credits  │ Avg Days    │   |
| ├──────────────────┼────────┼──────────┼─────────────┤   |
| │ Basic Monthly    │ 45     │ 2,800    │ 14 days left│   |
| │ Pro Monthly      │ 82     │ 19,000   │ 18 days left│   |
| │ Enterprise       │ 30     │ 85,000   │ 22 days left│   |
| │ Free Trial       │ 15     │ 500      │ 8 days left │   |
| └──────────────────┴────────┴──────────┴─────────────┘   |
|                                                           |
|                         [← Back]  [Next: Preview →]       |
+-----------------------------------------------------------+
```

### Livewire Data

```php
public string $migrationStrategy = 'immediate'; // 'immediate' | 'on_renewal'
```

---

## Adim 4: Onizleme & Calistir

**Amac:** Yapilacak tum degisiklikleri ozetle, onay al, calistir.

### Gorunum (calistirmadan once)

```
+-----------------------------------------------------------+
| Step 4: Preview & Execute                                 |
+-----------------------------------------------------------+
|                                                           |
| Review the migration summary before executing.            |
|                                                           |
| Plans to convert: 4                                       |
| ┌──────────────────┬───────────┬──────────────────────┐   |
| │ Plan             │ Amount    │ Status               │   |
| ├──────────────────┼───────────┼──────────────────────┤   |
| │ Basic Monthly    │ 2,800     │ ⏳ Pending           │   |
| │ Pro Monthly      │ 19,000    │ ⏳ Pending           │   |
| │ Enterprise       │ 85,000    │ ⏳ Pending           │   |
| │ Free Trial       │ 500       │ ⏳ Pending           │   |
| └──────────────────┴───────────┴──────────────────────┘   |
|                                                           |
| Users to migrate: 172                                     |
| Strategy: Immediate (pro-rata)                            |
|                                                           |
| ⚠ This action will:                                      |
|   • Convert 4 plans to shared credit system               |
|   • Migrate 172 users immediately                         |
|   • All changes are logged and reversible                 |
|                                                           |
|                     [← Back]  [🚀 Execute Migration]      |
+-----------------------------------------------------------+
```

### Gorunum (calistirdiktan sonra)

```
+-----------------------------------------------------------+
| Step 4: Preview & Execute                                 |
+-----------------------------------------------------------+
|                                                           |
| ✅ Migration completed successfully!                      |
|                                                           |
| Plans converted: 4/4                                      |
| ┌──────────────────┬───────────┬──────────────────────┐   |
| │ Plan             │ Amount    │ Status               │   |
| ├──────────────────┼───────────┼──────────────────────┤   |
| │ Basic Monthly    │ 2,800     │ ✅ Done              │   |
| │ Pro Monthly      │ 19,000    │ ✅ Done              │   |
| │ Enterprise       │ 85,000    │ ✅ Done              │   |
| │ Free Trial       │ 500       │ ✅ Done              │   |
| └──────────────────┴───────────┴──────────────────────┘   |
|                                                           |
| Users migrated: 172/172                                   |
| ┌──────────────────┬──────────┬──────────┬────────────┐  |
| │ Plan             │ Migrated │ Failed   │ Skipped    │  |
| ├──────────────────┼──────────┼──────────┼────────────┤  |
| │ Basic Monthly    │ 45       │ 0        │ 0          │  |
| │ Pro Monthly      │ 82       │ 0        │ 0          │  |
| │ Enterprise       │ 30       │ 0        │ 0          │  |
| │ Free Trial       │ 15       │ 0        │ 0          │  |
| └──────────────────┴──────────┴──────────┴────────────┘  |
|                                                           |
| [↩ Rollback All]              [Next: Notification →]      |
+-----------------------------------------------------------+
```

### Livewire Migration Metodu

```php
public bool $executed = false;
public array $results = [];

public function executeMigration(): void
{
    DB::beginTransaction();
    try {
        // 1. Planlari donustur
        foreach ($this->planConfigs as $config) {
            $plan = Plan::find($config['plan_id']);
            $plan->credit_system_type = 'shared';
            $plan->shared_credits_amount = $config['shared_credits_amount'];
            $plan->reset_credits_on_renewal = $config['reset_credits_on_renewal'];
            $plan->save();
        }

        // 2. Kullanicilari gecir (strategy'ye gore)
        if ($this->migrationStrategy === 'immediate') {
            $this->migrateUsersImmediate();
        }
        // 'on_renewal' seciliyse kullanicilara dokunma

        DB::commit();
        $this->executed = true;
    } catch (\Throwable $e) {
        DB::rollBack();
        $this->addError('migration', $e->getMessage());
    }
}

private function migrateUsersImmediate(): void
{
    $service = app(SharedCreditService::class);

    foreach ($this->selectedPlanIds as $planId) {
        $plan = Plan::find($planId);
        $users = User::whereHas('subscriptions', fn($q) =>
            $q->where('plan_id', $planId)
              ->whereIn('stripe_status', ['active', 'trialing', ...])
        )->where('credit_system_type', 'separated')->get();

        foreach ($users as $user) {
            $subscription = $user->subscriptions()
                ->where('plan_id', $planId)->latest()->first();

            // Pro-rata hesapla
            $totalDays = $subscription->created_at->diffInDays($subscription->ends_at) ?: 30;
            $remainingDays = max(0, now()->diffInDays($subscription->ends_at, false));
            $ratio = $remainingDays / $totalDays;

            $credits = round($plan->shared_credits_amount * $ratio, 4);

            // Kullaniciyi gecir
            $user->credit_system_type = 'shared';
            $user->shared_credits = $credits;
            $user->save();

            // Transaction logla
            $service->logTransaction(
                user: $user,
                actionType: 'migration',
                amount: $credits,
                balanceAfter: $credits,
                description: "Migrated from separated plan: {$plan->name}",
            );
        }
    }
}
```

---

## Adim 5: Bildirim (Opsiyonel)

**Amac:** Migrate edilen kullanicilara email bildirimi gonderme.

### Gorunum

```
+-----------------------------------------------------------+
| Step 5: Notification (Optional)                           |
+-----------------------------------------------------------+
|                                                           |
| Send email notification to migrated users?                |
|                                                           |
| ○ Skip notification                                       |
| ◉ Send notification via newsletter                        |
|                                                           |
| ┌───────────────────────────────────────────────────────┐ |
| │ Template: [Select or create template        ▾]        │ |
| │                                                       │ |
| │ Recipients: 172 migrated users                        │ |
| │ (credit_system_type = 'shared')                       │ |
| │                                                       │ |
| │ Preview:                                              │ |
| │ ┌─────────────────────────────────────────────────┐   │ |
| │ │ Subject: Your Credit System Has Been Updated    │   │ |
| │ │                                                 │   │ |
| │ │ Hi {user_name},                                 │   │ |
| │ │                                                 │   │ |
| │ │ Your credit system has been upgraded to the     │   │ |
| │ │ new Shared Credit system.                       │   │ |
| │ │                                                 │   │ |
| │ │ Your balance: {shared_credits} credits          │   │ |
| │ │ Plan: {plan_name}                               │   │ |
| │ └─────────────────────────────────────────────────┘   │ |
| └───────────────────────────────────────────────────────┘ |
|                                                           |
| [← Back]                              [Complete ✓]        |
+-----------------------------------------------------------+
```

### Newsletter Entegrasyonu

Mevcut newsletter sistemini kullanir:
- `EmailTemplates` modeli ile template secimi
- `SendEmail` job ile kuyruga alinir
- Yeni recipient filtresi: "Shared Credit Users"

---

## Rollback

Wizard'in 4. adiminda "Rollback All" butonu gosterilir (migration calistirildiktan sonra).

### Rollback Mantigi

```php
public function rollbackMigration(): void
{
    DB::beginTransaction();
    try {
        // 1. Kullanicilari geri al
        $transactions = SharedCreditTransaction::where('action_type', 'migration')->get();
        foreach ($transactions as $tx) {
            $user = User::find($tx->user_id);
            $user->credit_system_type = 'separated';
            $user->shared_credits = 0;
            $user->save();
            $tx->delete();
        }

        // 2. Planlari geri al
        foreach ($this->selectedPlanIds as $planId) {
            $plan = Plan::find($planId);
            $plan->credit_system_type = 'separated';
            // shared_credits_amount korunur (tekrar migration icin)
            $plan->save();
        }

        DB::commit();
        $this->executed = false;
    } catch (\Throwable $e) {
        DB::rollBack();
        $this->addError('rollback', $e->getMessage());
    }
}
```

---

## Veri Guvenligi

- `entity_credits` migration sirasinda ASLA silinmez (rollback icin korunur)
- `ai_models` planlarda ASLA silinmez (rollback icin korunur)
- Tum kredi degisiklikleri `shared_credit_transactions` tablosuna `action_type = 'migration'` ile loglanir
- `DB::beginTransaction()` ile atomik islem
- Rollback her zaman mumkun

---

## Dosya Yapisi

### Yeni Dosyalar

| Dosya | Aciklama |
|-------|---------|
| `app/Livewire/Admin/Finance/SharedCreditMigration.php` | Wizard Livewire component (5 adim) |
| `resources/views/livewire/admin/finance/shared-credit-migration.blade.php` | Ana wizard view |
| `resources/views/default/panel/admin/finance/shared-credit-migration/step-1-analysis.blade.php` | Analiz adimi |
| `resources/views/default/panel/admin/finance/shared-credit-migration/step-2-configure.blade.php` | Plan yapilandirma |
| `resources/views/default/panel/admin/finance/shared-credit-migration/step-3-users.blade.php` | Kullanici gecis secimi |
| `resources/views/default/panel/admin/finance/shared-credit-migration/step-4-execute.blade.php` | Onizleme & calistir |
| `resources/views/default/panel/admin/finance/shared-credit-migration/step-5-notification.blade.php` | Bildirim (opsiyonel) |

### Degistirilecek Dosyalar

| Dosya | Degisiklik |
|-------|-----------|
| `routes/panel.php` | Yeni route: `admin/finance/shared-credit-migration` |
| `app/Http/Controllers/Admin/Finance/` | Yeni controller veya mevcut'a route ekleme |
| Newsletter recipient filtresi | "Shared Credit Users" secenegi eklenir |

---

## Uygulama Sirasi

```
1. Livewire component + WithSteps trait → SharedCreditMigration.php
2. Step 1 view: Analiz (plan listesi + istatistikler)
3. Step 2 view: Plan yapilandirma (miktar hesaplama + unlimited model girisi)
4. Step 3 view: Kullanici gecis stratejisi secimi
5. Step 4 view: Onizleme + execute + rollback
6. Step 5 view: Newsletter bildirimi (opsiyonel)
7. Route + menu link ekleme
8. Newsletter filtresi ekleme
```

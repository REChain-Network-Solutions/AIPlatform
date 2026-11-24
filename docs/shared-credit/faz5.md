# Faz 5: Team Shared Credits

> **Durum:** Bekliyor
> **Bagimlilik:** Faz 4
> **Referans:** [shared-credit-pool-v2.md](../shared-credit-pool-v2.md) Bolum 3.2, 8.4

## Ozet

Takim bazli shared credit havuzu. Team manager bakiye yonetimi, member'lar team havuzundan harcama. User bakiyesi + team bakiyesi birlestirilir.

## Mevcut Team Yapisi

### Veritabani (Faz 1'de eklendi)
- `teams.credit_system_type` (default 'separated')
- `teams.shared_credits` (decimal 12,4)
- `Team::isSharedCreditTeam(): bool`

### Mevcut Kredi Birlestirme
`HasCreditLimit::getCredit()` (L52-88) user ve team kredilerini birlestiriyor:
```
Separated: user.entity_credits[engine][model].credit + team.entity_credits[engine][model].credit
```
Team member status kontrolu: sadece `active` member'lar team kredisini kullanabilir.

### Team View Dosyalari

| Dosya | Aciklama |
|-------|----------|
| `resources/views/default/panel/user/team/index.blade.php` | Team ana sayfa (overview, invite, members) |
| `resources/views/default/panel/user/team/partials/overview.blade.php` | Team genel bakis |
| `resources/views/default/panel/user/team/partials/team-members.blade.php` | Member listesi |
| `resources/views/default/panel/user/team/edit.blade.php` | Member duzenleme + credit list |

`edit.blade.php` icinde team kredi gosterimi: `<x-credit-list :team="$team" :showLegend="true" />`

## Adimlar

### 5.1: Team Shared Credit Birlestirme

**Dosya:** `app/Domains/Entity/Concerns/HasCreditLimit.php`

`getCredit()` metodundaki shared credit guard'ina team desteği:

```php
if ($this->isSharedCreditUser()) {
    $user = $this->getUserWithCredit();
    $balance = (float) $user->shared_credits;

    // Team shared credit birlestime
    $team = $this->team ?? $user->myTeam;
    if ($team?->exists && $team->isSharedCreditTeam()) {
        $memberStat = $user->teamMember;
        if (isset($memberStat) && $memberStat->status === 'active') {
            $balance += (float) $team->shared_credits;
        }
    }

    return [
        'credit'      => $balance,
        'isUnlimited' => false,
    ];
}
```

### 5.2: Team Havuzundan Dusme Logigi

**Dosya:** `app/Services/SharedCredit/SharedCreditService.php`

Yeni method: `deductWithTeam()` - once user bakiyesinden, yetersizse team bakiyesinden duser.

```php
public function deductWithTeam(
    User $user,
    Team $team,
    EntityEnum $entity,
    float $quantity,
    ?array $metadata = null
): ?SharedCreditTransaction {
    $unitCost = $this->resolveUnitCost($entity);
    $amount = $quantity * $unitCost;

    return DB::transaction(function () use ($user, $team, $entity, $amount, $unitCost, $quantity, $metadata) {
        $freshUser = User::query()->lockForUpdate()->find($user->id);
        $freshTeam = Team::query()->lockForUpdate()->find($team->id);

        $totalAvailable = $freshUser->shared_credits + $freshTeam->shared_credits;
        if ($totalAvailable < $amount) {
            return null; // Yetersiz toplam bakiye
        }

        $source = 'user';

        if ($freshUser->shared_credits >= $amount) {
            // User bakiyesinden tamamen karsilanir
            $freshUser->shared_credits -= $amount;
            $freshUser->save();
        } else {
            // User bakiyesi yetersiz, kalanini team'den al
            $remaining = $amount - $freshUser->shared_credits;
            $freshUser->shared_credits = 0;
            $freshUser->save();

            $freshTeam->shared_credits -= $remaining;
            $freshTeam->save();
            $source = 'team_split'; // kismen user, kismen team
        }

        return $this->logTransaction(
            user: $user,
            entity: $entity,
            actionType: 'deduct',
            amount: -$amount,
            balanceAfter: $freshUser->shared_credits,
            unitCost: $unitCost,
            quantity: $quantity,
            metadata: array_merge($metadata ?? [], ['source' => $source, 'team_id' => $team->id])
        );
    });
}

private function resolveUnitCost(EntityEnum $entity): float
{
    $override = $this->getCostOverride($entity->slug());
    return $override?->base_cost ?? $entity->sharedCreditIndex();
}
```

**HasCreditLimit::decreaseCredit()** guncelleme:

```php
// Shared credit guard icinde:
if ($this->isSharedCreditUser()) {
    $user = $this->getUser();
    $team = $this->team ?? $user->myTeam;
    $service = app(SharedCreditService::class);

    if ($team?->exists && $team->isSharedCreditTeam()) {
        $memberStat = $user->teamMember;
        if (isset($memberStat) && $memberStat->status === 'active') {
            $tx = $service->deductWithTeam($user, $team, $this->creditEnum(), $value);
            // ... UserUsageCredit log + total_spend guncelleme ...
            return $tx !== null;
        }
    }

    // Team yok veya aktif degil, sadece user'dan
    $tx = $service->deduct($user, $this->creditEnum(), $value);
    // ... log ...
    return $tx !== null;
}
```

### 5.3: Team Dashboard Guncelleme

**Dosya:** `resources/views/default/panel/user/team/partials/overview.blade.php`

```blade
@if($team->isSharedCreditTeam())
<x-card class="mt-4">
    <h4>{{ __('Team Shared Credit Pool') }}</h4>

    <div class="flex items-center gap-2">
        <x-tabler-coins class="w-6 h-6 text-primary" />
        <span class="text-3xl font-bold">{{ number_format($team->shared_credits, 0) }}</span>
        <span class="text-foreground/60">{{ __('credits') }}</span>
    </div>

    {{-- Member bazli harcama dagilimi (son 30 gun) --}}
    <h5 class="mt-4">{{ __('Usage by Member (last 30 days)') }}</h5>
    @php
        $memberUsage = \App\Models\SharedCreditTransaction::where('action_type', 'deduct')
            ->whereIn('user_id', $team->members->pluck('user_id'))
            ->where('created_at', '>=', now()->subDays(30))
            ->selectRaw('user_id, SUM(ABS(amount)) as total_usage')
            ->groupBy('user_id')
            ->get()
            ->keyBy('user_id');
    @endphp

    @foreach($team->members as $member)
    <div class="flex items-center justify-between py-2 border-b border-border last:border-0">
        <div class="flex items-center gap-2">
            <span>{{ $member->user->name }}</span>
            <x-badge variant="secondary" class="text-xs">{{ $member->role }}</x-badge>
        </div>
        <span class="font-medium">
            {{ number_format($memberUsage[$member->user_id]->total_usage ?? 0, 1) }}
            {{ __('credits used') }}
        </span>
    </div>
    @endforeach
</x-card>
@endif
```

### 5.4: Credit List Component - Team Shared Credit

**Dosya:** `resources/views/default/components/credit-list.blade.php`

Mevcut component `:team="$team"` prop'u destekliyor. Shared credit icin ek gosterim:

```blade
@if(auth()->user()?->isSharedCreditUser())
    {{-- User kendi bakiyesi --}}
    <div class="flex items-center gap-2">
        <span class="font-bold">{{ number_format(auth()->user()->shared_credits, 0) }}</span>
        <span class="text-xs text-foreground/60">{{ __('personal') }}</span>
    </div>

    {{-- Team bakiyesi (varsa) --}}
    @if($team?->isSharedCreditTeam())
    <div class="flex items-center gap-2 mt-1">
        <span class="font-bold">{{ number_format($team->shared_credits, 0) }}</span>
        <span class="text-xs text-foreground/60">{{ __('team pool') }}</span>
    </div>
    <div class="text-xs text-foreground/40 mt-1">
        {{ __('Total: :total credits', ['total' => number_format(auth()->user()->shared_credits + $team->shared_credits, 0)]) }}
    </div>
    @endif
@else
    {{-- Mevcut separated credit barlar --}}
@endif
```

### 5.5: Member Gunluk Limit (Opsiyonel)

Team manager'in member bazli gunluk harcama limiti koyabilmesi:

**Migration:**
```php
// team_members tablosuna yeni kolon
$table->decimal('daily_shared_credit_limit', 10, 4)->nullable();
// null = limitsiz
```

**SharedCreditService kontrol:**
```php
// deductWithTeam() icinde, dusme oncesi:
$memberStat = $user->teamMember;
if ($memberStat?->daily_shared_credit_limit) {
    $dailyUsage = SharedCreditTransaction::where('user_id', $user->id)
        ->where('action_type', 'deduct')
        ->whereDate('created_at', today())
        ->sum(DB::raw('ABS(amount)'));

    if (($dailyUsage + $amount) > $memberStat->daily_shared_credit_limit) {
        return null; // Gunluk limit asildi
    }
}
```

**Team Edit View:**
```blade
{{-- resources/views/default/panel/user/team/edit.blade.php --}}
@if($team->isSharedCreditTeam())
<x-forms.input
    type="number"
    name="daily_shared_credit_limit"
    step="0.01"
    :value="$member->daily_shared_credit_limit"
    :label="__('Daily Credit Limit (empty = unlimited)')"
    :placeholder="__('e.g. 500')"
/>
@endif
```

### 5.6: Team Plan Entegrasyonu

`is_team_plan` + `credit_system_type = shared` birlikte calisma:

**Dosya:** `app/Services/PaymentGateways/Contracts/CreditUpdater.php`

```php
// creditIncreaseSubscribePlan() icinde ek kontrol:
if ($plan->isSharedCreditPlan() && $plan->is_team_plan) {
    // Team plan: krediler team havuzuna eklenir (user yerine)
    $team = $user->myTeam ?? $user->ownedTeam;
    if ($team) {
        $amount = (float) $plan->shared_credits_amount;
        if ($plan->reset_credits_on_renewal) {
            $team->shared_credits = $amount;
        } else {
            $team->shared_credits += $amount;
        }
        $team->credit_system_type = 'shared';
        $team->save();

        // User'i da shared yap
        $user->credit_system_type = 'shared';
        $user->save();

        app(SharedCreditService::class)->logTransaction(
            user: $user,
            entity: null,
            actionType: 'subscription',
            amount: $amount,
            balanceAfter: $team->shared_credits,
            description: "Team plan: {$plan->name}"
        );
        return;
    }
}
```

## Dikkat Edilecekler

1. **Race Condition:** Team havuzundan birden fazla member ayni anda harcayabilir. `lockForUpdate()` ile hem user hem team satiri kilitlenir. Faz 5.2'deki `deductWithTeam()` bunu saglar.

2. **Dusme Onceligi:** User bakiyesi -> Team bakiyesi. Bu sirada her ikisi de `lockForUpdate()` ile kilitli olmali.

3. **Birlestirme Formulu:**
```
Separated: user.entity_credits[engine][model].credit + team.entity_credits[engine][model].credit
Shared:    user.shared_credits + team.shared_credits
```

4. **Team member status:** Sadece `active` member'lar team shared credit'ten faydalanabilir. `inactive` veya `pending` member'lar sadece kendi bakiyelerini kullanir.

## Ilgili Dosyalar

| Dosya | Islem | Detay |
|-------|-------|-------|
| `app/Domains/Entity/Concerns/HasCreditLimit.php` | MODIFY | getCredit() team guard, decreaseCredit() team dispatch |
| `app/Services/SharedCredit/SharedCreditService.php` | MODIFY | deductWithTeam(), resolveUnitCost() |
| `resources/views/default/panel/user/team/partials/overview.blade.php` | MODIFY | Team shared credit pool + member usage |
| `resources/views/default/panel/user/team/edit.blade.php` | MODIFY | daily_shared_credit_limit input |
| `resources/views/default/components/credit-list.blade.php` | MODIFY | Team shared credit gosterimi |
| `app/Services/PaymentGateways/Contracts/CreditUpdater.php` | MODIFY | Team plan subscribe |
| `database/migrations/` | CREATE | daily_shared_credit_limit kolonu |

## Test Plani

```
- [ ] Team member shared credit'ten harcama yapabiliyor mu (user bakiyesinden)
- [ ] User bakiyesi yetersiz -> team bakiyesinden otomatik dusme
- [ ] User + team bakiye birlestime dogru mu (getCredit)
- [ ] Toplam bakiye yetersiz oldugunda hata donuyor mu
- [ ] Inactive member team credit'ten faydalanamaz
- [ ] Gunluk member limit asilinca engelleniyor mu
- [ ] Gunluk limit null ise sinirsiz harcama
- [ ] Team plan subscribe: team.shared_credits artiyor mu (user yerine)
- [ ] Team plan: team.credit_system_type = 'shared' oluyor mu
- [ ] Race condition: concurrent team member harcamasi (lockForUpdate)
- [ ] Team dashboard: member bazli harcama dagilimi dogru
- [ ] Credit list: team shared credit bakiyesi gorunuyor mu
```

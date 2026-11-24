<?php

declare(strict_types=1);

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Services\SharedCredit\SharedCreditService;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class UserSharedCreditController extends Controller
{
    public function adjust(Request $request, User $user): RedirectResponse
    {
        $validated = $request->validate([
            'amount'      => 'required|numeric|min:0.01',
            'action'      => 'required|in:add,subtract,set',
            'description' => 'nullable|string|max:500',
        ]);

        $amount = (float) $validated['amount'];
        $description = $validated['description'] ?? __('Admin adjustment');
        $service = app(SharedCreditService::class);

        match ($validated['action']) {
            'add'      => $service->topUp($user, $amount, $description, null, 'admin_adjust'),
            'subtract' => $this->subtractCredits($user, $amount, $description, $service),
            'set'      => $this->setCredits($user, $amount, $description, $service),
        };

        return back()->with(['message' => __('Shared credits updated successfully.'), 'type' => 'success']);
    }

    public function updateType(Request $request, User $user): RedirectResponse
    {
        $validated = $request->validate([
            'credit_system_type' => 'required|in:separated,shared',
        ]);

        $user->update(['credit_system_type' => $validated['credit_system_type']]);

        return back()->with(['message' => __('Credit system type updated successfully.'), 'type' => 'success']);
    }

    private function subtractCredits(User $user, float $amount, string $description, SharedCreditService $service): void
    {
        DB::transaction(function () use ($user, $amount, $description, $service) {
            $freshUser = User::query()->lockForUpdate()->find($user->id);
            $freshUser->shared_credits = max(0, $freshUser->shared_credits - $amount);
            $freshUser->save();

            $service->logTransaction(
                user: $freshUser,
                actionType: 'admin_adjust',
                amount: -$amount,
                balanceAfter: $freshUser->shared_credits,
                description: $description,
            );
        });
    }

    private function setCredits(User $user, float $amount, string $description, SharedCreditService $service): void
    {
        DB::transaction(function () use ($user, $amount, $description, $service) {
            $freshUser = User::query()->lockForUpdate()->find($user->id);
            $previousBalance = $freshUser->shared_credits;
            $freshUser->shared_credits = $amount;
            $freshUser->save();

            $service->logTransaction(
                user: $freshUser,
                actionType: 'admin_adjust',
                amount: $amount - $previousBalance,
                balanceAfter: $amount,
                description: $description,
            );
        });
    }
}

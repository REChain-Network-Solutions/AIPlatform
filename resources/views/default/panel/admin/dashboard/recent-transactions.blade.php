@if (cache()->get('recent_transactions_enabled', false))
    <x-card
        class="flex flex-col"
        class:body="flex flex-col justify-center grow px-0"
        id="{{ 'admin-card-' . ($widget?->name?->value ?? 'recent-transactions') }}"
    >
        <x-slot:head
            class="flex items-center justify-between px-5 py-3.5"
        >
            <div class="flex items-center gap-4">
                <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                    <x-tabler-building-bank
                        class="size-6"
                        stroke-width="1.5"
                    />
                </x-lqd-icon>
                <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                    {{ __('Recent Transactions') }}
                    <x-info-tooltip text="{{ __('Latest payments made by your users.') }}" />
                </h4>
            </div>
            <x-button
                variant="link"
                href="{{ route('dashboard.admin.bank.transactions.list') }}"
            >
                <span class="text-nowrap font-bold text-foreground"> {{ __('View All') }} </span>
                <x-tabler-chevron-right class="ms-auto size-4" />
            </x-button>
        </x-slot:head>
        <div class="flex h-full flex-col gap-3">
            @foreach ($recentTransactions as $transaction)
                <div class="flex items-center justify-between border-b border-card-border px-6 pb-4 pt-2">
                    <div class="flex flex-col gap-1">
                        <p class="mb-0 text-lg font-semibold text-heading-foreground">
                            {{ $transaction->plan->name }}
                        </p>
                        <span
                            class="text-lg font-normal text-foreground/50">{{ $transaction->created_at->diffForHumans() }},
                            {{ ucwords(str_replace('_', ' ', $transaction->payment_type)) }}</span>
                    </div>
                    <div class="flex flex-col items-end">
                        <p class="mb-0 text-lg font-semibold text-heading-foreground">
                            <x-number-with-plus-minus :value="$transaction->price"></x-number-with-plus-minus>
                        </p>
                        @php
                            $status = $transaction->status == 'Success' ? 'Completed' : 'Pending';
                        @endphp
                        <x-transaction-status :status="$status" />
                    </div>
                </div>
            @endforeach
        </div>
    </x-card>
@endif

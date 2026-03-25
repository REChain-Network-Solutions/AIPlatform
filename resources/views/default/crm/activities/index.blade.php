@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Activities — CRM'))
@section('titlebar_title', __('Activities'))
@section('titlebar_subtitle', __('All logged calls, notes, tasks, and meetings.'))

@section('content')
    <div class="py-10">
        <form method="GET" class="mb-6 flex flex-wrap items-center gap-3">
            <select name="type" class="lqd-input lqd-input-md">
                <option value="">{{ __('All types') }}</option>
                @foreach (['note', 'call', 'email', 'meeting', 'task', 'deadline'] as $t)
                    <option value="{{ $t }}" @selected(request('type') === $t)>{{ ucfirst($t) }}</option>
                @endforeach
            </select>
            <label class="flex items-center gap-2 text-sm">
                <input type="checkbox" name="overdue" value="1" @checked(request('overdue')) class="rounded border-border">
                {{ __('Overdue only') }}
            </label>
            <label class="flex items-center gap-2 text-sm">
                <input type="checkbox" name="pending" value="1" @checked(request('pending')) class="rounded border-border">
                {{ __('Pending only') }}
            </label>
            <x-button type="submit" size="sm" variant="outline">{{ __('Filter') }}</x-button>
        </form>

        <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
            <table class="w-full text-sm">
                <thead class="border-b border-border bg-muted/40 text-left text-xs font-medium text-muted-foreground">
                    <tr>
                        <th class="px-4 py-3">{{ __('Type') }}</th>
                        <th class="px-4 py-3">{{ __('Subject') }}</th>
                        <th class="px-4 py-3">{{ __('Linked to') }}</th>
                        <th class="px-4 py-3">{{ __('Due') }}</th>
                        <th class="px-4 py-3">{{ __('Status') }}</th>
                        <th class="px-4 py-3 text-right">{{ __('Actions') }}</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-border">
                    @forelse ($activities as $activity)
                        <tr class="hover:bg-accent/30">
                            <td class="px-4 py-3">
                                <div class="flex items-center gap-2 text-muted-foreground">
                                    <x-dynamic-component :component="$activity->type_icon" class="w-4" />
                                    <span class="capitalize">{{ $activity->type }}</span>
                                </div>
                            </td>
                            <td class="px-4 py-3">
                                <p class="font-medium text-heading-foreground">{{ $activity->subject }}</p>
                                @if ($activity->body)<p class="text-xs text-muted-foreground">{{ Str::limit($activity->body, 60) }}</p>@endif
                            </td>
                            <td class="px-4 py-3 text-xs text-muted-foreground">
                                @if ($activity->contact)<a href="{{ route('dashboard.crm.contacts.show', $activity->contact) }}" class="hover:text-primary hover:underline">{{ $activity->contact->full_name }}</a>@endif
                                @if ($activity->deal)<a href="{{ route('dashboard.crm.deals.show', $activity->deal) }}" class="hover:text-primary hover:underline">{{ $activity->deal->title }}</a>@endif
                                @if (!$activity->contact && !$activity->deal)—@endif
                            </td>
                            <td class="px-4 py-3 text-xs">
                                @if ($activity->due_at)
                                    <span @class(['text-red-500 font-medium' => $activity->due_at->isPast() && !$activity->is_done, 'text-muted-foreground' => !($activity->due_at->isPast() && !$activity->is_done)])>
                                        {{ $activity->due_at->format('M d, Y H:i') }}
                                    </span>
                                @else
                                    <span class="text-muted-foreground">—</span>
                                @endif
                            </td>
                            <td class="px-4 py-3">
                                @if ($activity->is_done)
                                    <span class="rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700">{{ __('Done') }}</span>
                                @else
                                    <span class="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700">{{ __('Pending') }}</span>
                                @endif
                            </td>
                            <td class="px-4 py-3 text-right">
                                <form method="POST" action="{{ route('dashboard.crm.activities.destroy', $activity) }}" onsubmit="return confirm('{{ __('Delete?') }}')">
                                    @csrf @method('DELETE')
                                    <button type="submit" class="text-muted-foreground hover:text-red-500"><x-tabler-trash class="w-4" /></button>
                                </form>
                            </td>
                        </tr>
                    @empty
                        <tr>
                            <td colspan="6" class="py-16 text-center text-muted-foreground">
                                <x-tabler-activity class="mx-auto mb-3 w-10 opacity-30" />
                                <p>{{ __('No activities found.') }}</p>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>
        <div class="mt-4">{{ $activities->links() }}</div>
    </div>
@endsection

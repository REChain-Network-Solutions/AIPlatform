<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmActivity;
use App\Models\Crm\CrmCompany;
use App\Models\Crm\CrmContact;
use App\Models\Crm\CrmDeal;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmContactController extends Controller
{
    public function index(Request $request): View
    {
        $query = CrmContact::forUser(Auth::id())
            ->with('company')
            ->latest();

        if ($search = $request->input('search')) {
            $query->where(function ($q) use ($search) {
                $q->where('first_name', 'like', "%{$search}%")
                    ->orWhere('last_name', 'like', "%{$search}%")
                    ->orWhere('email', 'like', "%{$search}%")
                    ->orWhere('company_name', 'like', "%{$search}%");
            });
        }

        if ($status = $request->input('status')) {
            $query->where('status', $status);
        }

        $contacts = $query->paginate(20)->withQueryString();

        return view('crm.contacts.index', compact('contacts'));
    }

    public function create(): View
    {
        $companies = CrmCompany::forUser(Auth::id())->orderBy('name')->get();

        return view('crm.contacts.create', compact('companies'));
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'first_name'     => 'required|string|max:100',
            'last_name'      => 'nullable|string|max:100',
            'email'          => 'nullable|email|max:255',
            'phone'          => 'nullable|string|max:30',
            'job_title'      => 'nullable|string|max:100',
            'company_name'   => 'nullable|string|max:255',
            'crm_company_id' => 'nullable|exists:crm_companies,id',
            'linkedin_url'   => 'nullable|url|max:255',
            'website'        => 'nullable|url|max:255',
            'address'        => 'nullable|string|max:500',
            'city'           => 'nullable|string|max:100',
            'state'          => 'nullable|string|max:100',
            'country'        => 'nullable|string|size:2',
            'status'         => 'required|in:lead,prospect,customer,churned,partner',
            'notes'          => 'nullable|string',
        ]);

        $contact = CrmContact::create(array_merge($validated, ['user_id' => Auth::id()]));

        return redirect()
            ->route('dashboard.crm.contacts.show', $contact)
            ->with('success', __('Contact created successfully.'));
    }

    public function show(CrmContact $contact): View
    {
        $this->authorizeContact($contact);

        $contact->load(['company', 'deals.stage', 'activities' => fn ($q) => $q->latest()->limit(20)]);

        $openDealsCount  = $contact->deals()->where('status', 'open')->count();
        $totalDealValue  = $contact->deals()->where('status', 'open')->sum('value');

        return view('crm.contacts.show', compact('contact', 'openDealsCount', 'totalDealValue'));
    }

    public function edit(CrmContact $contact): View
    {
        $this->authorizeContact($contact);
        $companies = CrmCompany::forUser(Auth::id())->orderBy('name')->get();

        return view('crm.contacts.edit', compact('contact', 'companies'));
    }

    public function update(Request $request, CrmContact $contact): RedirectResponse
    {
        $this->authorizeContact($contact);

        $validated = $request->validate([
            'first_name'     => 'required|string|max:100',
            'last_name'      => 'nullable|string|max:100',
            'email'          => 'nullable|email|max:255',
            'phone'          => 'nullable|string|max:30',
            'job_title'      => 'nullable|string|max:100',
            'company_name'   => 'nullable|string|max:255',
            'crm_company_id' => 'nullable|exists:crm_companies,id',
            'linkedin_url'   => 'nullable|url|max:255',
            'website'        => 'nullable|url|max:255',
            'address'        => 'nullable|string|max:500',
            'city'           => 'nullable|string|max:100',
            'state'          => 'nullable|string|max:100',
            'country'        => 'nullable|string|size:2',
            'status'         => 'required|in:lead,prospect,customer,churned,partner',
            'notes'          => 'nullable|string',
        ]);

        $contact->update($validated);

        return redirect()
            ->route('dashboard.crm.contacts.show', $contact)
            ->with('success', __('Contact updated successfully.'));
    }

    public function destroy(CrmContact $contact): RedirectResponse
    {
        $this->authorizeContact($contact);
        $contact->delete();

        return redirect()
            ->route('dashboard.crm.contacts.index')
            ->with('success', __('Contact deleted.'));
    }

    private function authorizeContact(CrmContact $contact): void
    {
        abort_unless($contact->user_id === Auth::id(), 403);
    }
}

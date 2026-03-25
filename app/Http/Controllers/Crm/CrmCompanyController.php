<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmCompany;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmCompanyController extends Controller
{
    public function index(Request $request): View
    {
        $query = CrmCompany::forUser(Auth::id())->latest();

        if ($search = $request->input('search')) {
            $query->where(function ($q) use ($search) {
                $q->where('name', 'like', "%{$search}%")
                    ->orWhere('domain', 'like', "%{$search}%")
                    ->orWhere('industry', 'like', "%{$search}%");
            });
        }

        $companies = $query->withCount('contacts', 'deals')->paginate(20)->withQueryString();

        return view('crm.companies.index', compact('companies'));
    }

    public function create(): View
    {
        return view('crm.companies.create');
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'name'           => 'required|string|max:255',
            'domain'         => 'nullable|string|max:255',
            'website'        => 'nullable|url|max:255',
            'phone'          => 'nullable|string|max:30',
            'industry'       => 'nullable|string|max:100',
            'type'           => 'nullable|string|max:50',
            'employee_count' => 'nullable|integer|min:0',
            'annual_revenue' => 'nullable|numeric|min:0',
            'address'        => 'nullable|string|max:500',
            'city'           => 'nullable|string|max:100',
            'state'          => 'nullable|string|max:100',
            'country'        => 'nullable|string|size:2',
            'description'    => 'nullable|string',
        ]);

        $company = CrmCompany::create(array_merge($validated, ['user_id' => Auth::id()]));

        return redirect()
            ->route('dashboard.crm.companies.show', $company)
            ->with('success', __('Company created successfully.'));
    }

    public function show(CrmCompany $company): View
    {
        $this->authorizeCompany($company);
        $company->load(['contacts', 'deals.stage', 'activities' => fn ($q) => $q->latest()->limit(20)]);

        return view('crm.companies.show', compact('company'));
    }

    public function edit(CrmCompany $company): View
    {
        $this->authorizeCompany($company);

        return view('crm.companies.edit', compact('company'));
    }

    public function update(Request $request, CrmCompany $company): RedirectResponse
    {
        $this->authorizeCompany($company);

        $validated = $request->validate([
            'name'           => 'required|string|max:255',
            'domain'         => 'nullable|string|max:255',
            'website'        => 'nullable|url|max:255',
            'phone'          => 'nullable|string|max:30',
            'industry'       => 'nullable|string|max:100',
            'type'           => 'nullable|string|max:50',
            'employee_count' => 'nullable|integer|min:0',
            'annual_revenue' => 'nullable|numeric|min:0',
            'address'        => 'nullable|string|max:500',
            'city'           => 'nullable|string|max:100',
            'state'          => 'nullable|string|max:100',
            'country'        => 'nullable|string|size:2',
            'description'    => 'nullable|string',
        ]);

        $company->update($validated);

        return redirect()
            ->route('dashboard.crm.companies.show', $company)
            ->with('success', __('Company updated successfully.'));
    }

    public function destroy(CrmCompany $company): RedirectResponse
    {
        $this->authorizeCompany($company);
        $company->delete();

        return redirect()
            ->route('dashboard.crm.companies.index')
            ->with('success', __('Company deleted.'));
    }

    private function authorizeCompany(CrmCompany $company): void
    {
        abort_unless($company->user_id === Auth::id(), 403);
    }
}

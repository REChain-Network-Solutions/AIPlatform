<?php

namespace Modules\EInvoice\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use App\Models\ClientDetails;
use App\Models\Invoice;
use App\Models\Project;
use App\Models\User;
use Illuminate\Http\Request;
use Modules\EInvoice\DataTables\InvoicesDataTable;
use Modules\EInvoice\Entities\EInvoiceCompanySetting;
use Modules\EInvoice\Entities\EInvoiceDraft;
use Modules\EInvoice\AI\ClientInterface;
use Modules\EInvoice\Jobs\GenerateInvoiceNote;
use Modules\EInvoice\Helper\InvoiceXmlGenerate;
use Modules\EInvoice\Entities\Invoice as ModInvoice;
use Modules\EInvoice\Entities\InvoiceItem as ModInvoiceItem;
use Saloon\XmlWrangler\Data\RootElement;
use Saloon\XmlWrangler\XmlWriter;

class EInvoiceController extends AccountBaseController
{
    public function aiSettings()
    {
        $this->pageTitle = 'E-Invoice AI Settings';
        return view('einvoice::settings.ai', $this->data);
    }

    public function aiTest(\Illuminate\Http\Request $request)
    {
        /** @var ClientInterface $ai */
        $ai = app(ClientInterface::class);
        $out = $ai->complete($request->input('prompt', 'Say hello from EInvoice AI test.'));
        return response()->json(['ok' => true, 'output' => $out]);
    }

    public function aiHealth()
    {
        /** @var \\Modules\\EInvoice\\AI\\ClientInterface $ai */
        $ai = app(\\Modules\\EInvoice\\AI\\ClientInterface::class);
        return response()->json($ai->health());
    }

    
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'einvoice::app.menu.einvoice';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('invoices', $this->user->modules));

            return $next($request);
        });
    }

    public function index(InvoicesDataTable $dataTable)
    {
        $viewPermission = user()->permission('view_invoices');
        abort_403(!in_array($viewPermission, ['all', 'added', 'owned', 'both']));

        if (!request()->ajax()) {
            $this->projects = Project::allProjects();

            if (in_array('client', user_roles())) {
                $this->clients = User::client();
            }
            else {
                $this->clients = User::allClients();
            }
        }

        return $dataTable->render('einvoice::index', $this->data);
    }

    public function exportXml($id)
    {
        $viewPermission = user()->permission('view_invoices');
        abort_403(!in_array($viewPermission, ['all', 'added', 'owned', 'both']));

        $invoice = Invoice::with('client')->findOrFail($id);

        if (!$invoice->client?->clientdetails?->electronic_address || !$invoice->client?->clientdetails?->electronic_address_scheme) {
            return redirect()->route('einvoice.index')->with('message', __('einvoice::app.clientElectronicAddressNotSet'));
        }

        $array = [];

        $writer = new XmlWriter();

        $rootElement = RootElement::make('Invoice', attributes: [
            'xmlns' => 'urn:oasis:names:specification:ubl:schema:xsd:Invoice-2',
            'xmlns:cbc' => 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
            'xmlns:cac' => 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2',
        ]);

        $array = InvoiceXmlGenerate::generateXml($invoice);

        $xml = $writer->write($rootElement, $array);

        return response()->streamDownload(function () use ($xml) {
            echo $xml;
        }, $invoice->invoice_number . '.xml');
    }

    public function settings()
    {
        abort_403(user()->permission('manage_finance_setting') != 'all');

        $this->activeSettingMenu = 'einvoice_settings';
        $this->pageTitle = 'einvoice::app.menu.einvoiceSettings';

        return view('einvoice::settings.index', $this->data);
    }

    public function settingsModal()
    {
        abort_403(user()->permission('manage_finance_setting') != 'all');

        return view('einvoice::settings.modal');
    }

    public function saveSettings(Request $request)
    {
        abort_403(user()->permission('manage_finance_setting') != 'all');

        EInvoiceCompanySetting::updateOrCreate(
            ['company_id' => company()->id],
            [
                'electronic_address' => $request->electronic_address,
                'electronic_address_scheme' => $request->electronic_address_scheme,
                'e_invoice_company_id' => $request->e_invoice_company_id,
                'e_invoice_company_id_scheme' => $request->e_invoice_company_id_scheme,
            ]
        );

        return Reply::success(__('messages.updateSuccess'));
    }

    public function clientModal($id)
    {
        $this->clientDetails = ClientDetails::findOrFail($id);

        return view('einvoice::client.modal', $this->data);
    }

    public function clientSave(Request $request, $id)
    {
        $clientDetails = ClientDetails::findOrFail($id);
        $clientDetails->electronic_address = $request->electronic_address;
        $clientDetails->electronic_address_scheme = $request->electronic_address_scheme;
        $clientDetails->saveQuietly();

        return Reply::success(__('messages.updateSuccess'));
    
    public function generateNote($invoiceId)
    {
        $viewPermission = user()->permission('view_invoices');
        abort_403(!in_array($viewPermission, ['all', 'added', 'owned', 'both']));

        GenerateInvoiceNote::dispatch((int)$invoiceId, optional(user())->id);

        return response()->json(['ok' => true, 'queued' => true]);
    
    public function latestNote($invoiceId)
    {
        $viewPermission = user()->permission('view_invoices');
        abort_403(!in_array($viewPermission, ['all', 'added', 'owned', 'both']));

        $note = \Modules\EInvoice\Entities\EInvoiceNote::where('invoice_id', (int)$invoiceId)
            ->orderByDesc('updated_at')
            ->orderByDesc('created_at')
            ->first();

        if (!$note) {
            return response()->json(['note' => null], 404);
        }
        return response()->json(['note' => $note]);
    
    /**
     * Build an invoice DRAFT from a natural-language prompt.
     * Returns JSON with structured items; stores a draft row.
     */
    public function aiInvoiceDraft(\Illuminate\Http\Request $request)
    {
        $perm = user()->permission('add_invoices') ?? user()->permission('create_invoices');
        abort_403(!in_array($perm, ['all', 'added', 'owned', 'both', 'yes', 'allow']));

        $prompt = trim($request->input('prompt', ''));
        $clientId = $request->input('client_id');

        /** @var \Modules\EInvoice\AI\ClientInterface $ai */
        $ai = app(\Modules\EInvoice\AI\ClientInterface::class);

        $schema = json_encode([
            'client_id' => 'int?',
            'currency' => 'string (ISO), e.g., USD',
            'due_days' => 'int (days from today)',
            'notes' => 'string',
            'items' => [
                ['description' => 'string', 'qty' => 'number', 'unit_price' => 'number']
            ]
        ]);

        $sys = 'You are a finance assistant. Output ONLY valid JSON matching the schema.';
        $user = 'Prompt: ' . $prompt . ' | Schema: ' . $schema;

        $raw = $ai->complete($user, ['system' => $sys, 'temperature' => 0.2, 'max_tokens' => 400]);
        $data = json_decode($raw, true);

        if (!is_array($data) || !isset($data['items']) || !is_array($data['items'])) {
            return response()->json(['ok' => false, 'error' => 'AI did not return valid JSON', 'raw' => $raw], 422);
        }

        if ($clientId && empty($data['client_id'])) {
            $data['client_id'] = (int)$clientId;
        }

        $draft = EInvoiceDraft::create([
            'user_id' => optional(user())->id,
            'client_id' => $data['client_id'] ?? null,
            'payload' => $data,
        ]);

        return response()->json(['ok' => true, 'draft_id' => $draft->id, 'draft' => $draft->payload]);
    }

    /**
     * Try to create a real Invoice from a DRAFT (best-effort, schema tolerant).
     * If Invoice model creation fails, returns the draft so the user can copy-paste.
     */
    public function createInvoiceFromDraft(\Illuminate\Http\Request $request, $draftId)
    {
        $perm = user()->permission('add_invoices') ?? user()->permission('create_invoices');
        abort_403(!in_array($perm, ['all', 'added', 'owned', 'both', 'yes', 'allow']));

        $draft = EInvoiceDraft::findOrFail((int)$draftId);
        $data = $draft->payload ?: [];

        try {
            // Minimal creation; adapt to your app's Invoice schema
            $invoice = new \App\Models\Invoice();
            if (isset($data['client_id'])) $invoice->client_id = (int)$data['client_id'];
            if (isset($data['currency'])) $invoice->currency = $data['currency'];
            if (isset($data['notes'])) $invoice->note = $data['notes'];
            // Set status draft if exists
            if (property_exists($invoice, 'status')) $invoice->status = 'draft';
            // Save basic invoice
            $invoice->save();

            // If your app has invoice_items table/model, insert items here.
            // (Left out intentionally to avoid guessing app-specific schema.)

            return response()->json(['ok' => true, 'invoice_id' => $invoice->id]);
        } catch (\Throwable $e) {
            return response()->json([
                'ok' => false,
                'error' => 'Could not create invoice automatically: ' . $e->getMessage(),
                'draft' => $data
            ], 200);
        }
    }
}

<?php
namespace Modules\Treasury\Services;
use Modules\Treasury\Contracts\LedgerPosterInterface;

class AccountingLedgerPoster implements LedgerPosterInterface
{
    protected $ledger;
    public function __construct()
    {
        // Resolve lazily to avoid hard dependency when Accounting not installed yet
        if (app()->bound('Modules\\Accounting\\Contracts\\LedgerInterface')) {
            $this->ledger = app('Modules\\Accounting\\Contracts\\LedgerInterface');
        } else {
            $this->ledger = null;
        }
    }

    public function postBankFees(array $data): ?string
    {
        if (!$this->ledger) return null;
        $entry = [
            'date' => $data['date'] ?? date('Y-m-d'),
            'memo' => 'Bank fees',
            'lines' => [
                ['account' => $data['expense_account'] ?? '6xxx', 'debit' => $data['amount'] ?? 0],
                ['account' => $data['bank_account_code'] ?? '1xxx', 'credit' => $data['amount'] ?? 0],
            ],
        ];
        return $this->ledger->post($entry);
    }

    public function postPayments(array $data): ?string
    {
        if (!$this->ledger) return null;
        $lines = [];
        foreach ($data['lines'] ?? [] as $ln) {
            $lines[] = ['account' => $ln['ap_account'] ?? '2xxx', 'debit' => $ln['amount'] ?? 0];
        }
        $lines[] = ['account' => $data['bank_account_code'] ?? '1xxx', 'credit' => array_reduce($data['lines'] ?? [], fn($c,$l)=>$c+($l['amount']??0), 0)];
        $entry = ['date'=>$data['date'] ?? date('Y-m-d'),'memo'=>'Payment run','lines'=>$lines];
        return $this->ledger->post($entry);
    }

    public function postInterest(array $data): ?string
    {
        if (!$this->ledger) return null;
        $amount = (float) ($data['amount'] ?? 0);
        $income = $amount >= 0;
        $entry = [
            'date' => $data['date'] ?? date('Y-m-d'),
            'memo' => 'Bank interest',
            'lines' => $income ? [
                ['account' => $data['bank_account_code'] ?? '1xxx', 'debit' => $amount],
                ['account' => $data['interest_income_account'] ?? '4xxx', 'credit' => $amount],
            ] : [
                ['account' => $data['interest_expense_account'] ?? '6xxx', 'debit' => abs($amount)],
                ['account' => $data['bank_account_code'] ?? '1xxx', 'credit' => abs($amount)],
            ],
        ];
        return $this->ledger->post($entry);
    }
}

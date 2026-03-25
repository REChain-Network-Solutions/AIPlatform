<?php
namespace Modules\Treasury\Console\Commands;
use Illuminate\Console\Command;
use Modules\Treasury\Models\PaymentRun;
use Modules\Treasury\Contracts\LedgerPosterInterface;

class PostPaymentRun extends Command
{
    protected $signature = 'treasury:post-payment-run {run_id}';
    protected $description = 'Post a payment run to the General Ledger (if Accounting is present).';
    public function handle(): int
    {
        $id = (int) $this->argument('run_id');
        $run = PaymentRun::with('lines')->findOrFail($id);
        $poster = app(LedgerPosterInterface::class);
        $lines = [];
        foreach ($run->lines as $ln) {
            $lines[] = ['ap_account' => '2xxx', 'amount' => (float)$ln->amount];
        }
        $jid = $poster->postPayments(['date'=>$run->scheduled_on, 'bank_account_code'=>'1xxx', 'lines'=>$lines]);
        if ($jid) { $run->posted_journal_id = $jid; $run->status = 'posted'; $run->save(); }
        $this->info('Posted run #'.$id.' journal='.($jid ?? 'none'));
        return self::SUCCESS;
    }
}

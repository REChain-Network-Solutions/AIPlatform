<?php
namespace Modules\Compliance\Services;
use AICore\Contracts\ClientInterface; use Illuminate\Support\Str;
class AiClient {
  public function __construct(private ClientInterface $core) {} 
  protected function chat(string $slug, array $ctx): string {
    $messages = [
      ['role'=>'system','content'=>'You are a Compliance assistant. Be precise and compliant.'],
      ['role'=>'user','content'=> json_encode(['prompt'=>$slug,'context'=>$ctx], JSON_UNESCAPED_SLASHES)],
    ];
    $opts = [
      'tenant_id' => $ctx['tenant_id'] ?? null,
      'feature'   => 'compliance',
      'purpose'   => $slug,
      'headers'   => ['X-Idempotency-Key' => (string) Str::uuid()],
    ];
    return $this->core->chat($messages, $opts);
  }
}

<?php
namespace Modules\Treasury\Contracts;
interface BankFeedImporterInterface {
  /** @return array<int, array{date:string, description:string, amount:float, reference?:string}> */
  public function parse(string $raw): array;
}

<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Invoice;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель InvoiceCancellationDetails.
 *
 * Комментарий к статусу ~`canceled`: кто отменил счет и по какой причине.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $party Участник процесса, который принял решение об отмене счета.
 * @property string $reason Причина отмены счета.
 */
class InvoiceCancellationDetails extends AbstractObject
{
    /**
     * Участник процесса, который принял решение об отмене счета.
     *
     * Возможные значения:
     * - `merchant` — продавец товаров и услуг (вы);
     * - `yoo_money` — ЮKassa.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [InvoiceCancellationDetailsPartyCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_party = null;

    /**
     * Причина отмены счета.
     *
     * Возможные значения:
     * - `invoice_canceled` — [счет отменен вручную](https://yookassa.ru/docs/support/merchant/invoices-to-clients/invoicing#invoicing__cancel) из личного кабинета ЮKassa;
     * - `invoice_expired` — истек срок действия счета, который вы установили в запросе на создание счета в параметре `expires_at`, и по счету нет ни одного успешного платежа;
     * - `general_decline` — причина не детализирована, поэтому пользователю следует обратиться к инициатору отмены счета за уточнением подробностей;
     * - `payment_canceled` — [платеж отменен по API](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#cancel) при оплате в две стадии;
     * - `payment_expired_on_capture` — [истек срок списания оплаты](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#hold) для платежа в две стадии.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [InvoiceCancellationDetailsReasonCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_reason = null;

    /**
     * Возвращает party.
     *
     * @return string|null
     */
    public function getParty(): ?string
    {
        return $this->_party;
    }

    /**
     * Устанавливает party.
     *
     * @param string|null $party Участник процесса, который принял решение об отмене счета. Возможные значения:  * ~`merchant` — продавец товаров и услуг (вы); * ~`yoo_money` — ЮKassa.
     *
     * @return self
     */
    public function setParty(?string $party = null): self
    {
        $this->_party = $this->validatePropertyValue('_party', $party);
        return $this;
    }

    /**
     * Возвращает reason.
     *
     * @return string|null
     */
    public function getReason(): ?string
    {
        return $this->_reason;
    }

    /**
     * Устанавливает reason.
     *
     * @param string|null $reason Причина отмены счета. Возможные значения:  * ~`invoice_canceled` — %[счет отменен вручную](/docs/support/merchant/invoices-to-clients/invoicing#invoicing__cancel) из личного кабинета ЮKassa; * ~`invoice_expired` — истек срок действия счета, который вы установили в запросе на создание счета в параметре `expires_at`, и по счету нет ни одного успешного платежа; * ~`general_decline` — причина не детализирована, поэтому пользователю следует обратиться к инициатору отмены счета за уточнением подробностей; * ~`payment_canceled` — %[платеж отменен по API](/developers/payment-acceptance/getting-started/payment-process#cancel) при оплате в две стадии; * ~`payment_expired_on_capture` — %[истек срок списания оплаты](/developers/payment-acceptance/getting-started/payment-process#hold) для платежа в две стадии.
     *
     * @return self
     */
    public function setReason(?string $reason = null): self
    {
        $this->_reason = $this->validatePropertyValue('_reason', $reason);
        return $this;
    }

}


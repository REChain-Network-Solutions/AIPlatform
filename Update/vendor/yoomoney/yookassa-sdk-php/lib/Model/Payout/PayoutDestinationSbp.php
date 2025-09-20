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

namespace YooKassa\Model\Payout;

use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PayoutToSbpDestination.
 *
 * Данные для выплаты через СБП на счет в банке или платежном сервисе.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $phone Телефон, к которому привязан счет получателя выплаты в системе участника СБП
 * @property string $bank_id Идентификатор выбранного участника СБП — банка или платежного сервиса, подключенного к сервису
 * @property string $bankId Идентификатор выбранного участника СБП — банка или платежного сервиса, подключенного к сервису
 * @property string $recipient_checked Проверка получателя выплаты
 * @property string $recipientChecked Проверка получателя выплаты
 */
class PayoutDestinationSbp extends AbstractPayoutDestination
{
    /** @var int Максимальная длина строки id банка. */
    public const MAX_LENGTH_BANK_ID = 12;

    /**
     * Телефон, к которому привязан счет получателя выплаты в системе участника СБП.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{4,15}$/")]
    private ?string $_phone = null;

    /**
     * Идентификатор участника СБП — банка или платежного сервиса, подключенного к сервису.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_BANK_ID)]
    private ?string $_bank_id = null;

    /**
     * Проверка получателя выплаты
     * * true — выплата проходила с проверкой получателя,
     * * false — выплата проходила без проверки получателя.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private ?bool $_recipient_checked = false;

    /**
     * Конструктор PayoutDestinationSbp.
     *
     * @param array|null $data
     */
    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::SBP);
    }

    /**
     * Возвращает телефон, к которому привязан счет получателя выплаты в системе участника СБП.
     *
     * @return string|null Телефон, к которому привязан счет получателя выплаты в системе участника СБП
     */
    public function getPhone(): ?string
    {
        return $this->_phone;
    }

    /**
     * Устанавливает телефон, к которому привязан счет получателя выплаты в системе участника СБП.
     *
     * @param string|null $phone Телефон, к которому привязан счет получателя выплаты в системе участника СБП
     *
     * @return self
     */
    public function setPhone(?string $phone = null): self
    {
        $this->_phone = $this->validatePropertyValue('_phone', $phone);
        return $this;
    }

    /**
     * Возвращает идентификатор выбранного участника СБП.
     *
     * @return string|null Идентификатор выбранного участника СБП
     */
    public function getBankId(): ?string
    {
        return $this->_bank_id;
    }

    /**
     * Устанавливает идентификатор выбранного участника СБП.
     *
     * @param string|null $bank_id Идентификатор выбранного участника СБП
     *
     * @return self
     */
    public function setBankId(?string $bank_id = null): self
    {
        $this->_bank_id = $this->validatePropertyValue('_bank_id', $bank_id);
        return $this;
    }

    /**
     * Возвращает признак проверки получателя выплаты.
     *
     * @return bool|null Признак проверки получателя выплаты
     */
    public function getRecipientChecked(): ?bool
    {
        return $this->_recipient_checked;
    }

    /**
     * Устанавливает признак проверки получателя выплаты.
     *
     * @param bool|null $recipient_checked Признак проверки получателя выплаты
     *
     * @return self
     */
    public function setRecipientChecked(?bool $recipient_checked = null): self
    {
        $this->_recipient_checked = $this->validatePropertyValue('_recipient_checked', $recipient_checked);
        return $this;
    }
}

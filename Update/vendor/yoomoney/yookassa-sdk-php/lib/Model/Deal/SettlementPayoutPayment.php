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

namespace YooKassa\Model\Deal;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\SettlementInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SettlementPayoutPayment.
 *
 * Данные о распределении денег.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $type Вид оплаты в чеке
 * @property AmountInterface $amount Размер оплаты
 */
class SettlementPayoutPayment extends AbstractObject implements SettlementInterface
{
    /**
     * Тип операции (payout - выплата продавцу)
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [SettlementPayoutPaymentType::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_type = null;

    /**
     * Размер оплаты
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Возвращает тип операции (payout - выплата продавцу).
     *
     * @return string|null Тип операции
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает вид оплаты в чеке.
     *
     * @param string|null $type
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает размер оплаты.
     *
     * @return AmountInterface|null Размер оплаты
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму платежа.
     *
     * @param AmountInterface|array|null $amount Сумма платежа
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }
}

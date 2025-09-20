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

namespace YooKassa\Model\Payment\PaymentMethod;

use DateTime;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodSberLoan.
 *
 * Оплата с использованием Кредита от СберБанка.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $loan_option Тариф кредита
 * @property string $loanOption Тариф кредита
 * @property AmountInterface $discount_amount Сумма скидки для рассрочки
 * @property AmountInterface $discountAmount Сумма скидки для рассрочки
 * @property DateTime $suspended_until Время, когда заканчивается период охлаждения кредита или рассрочки.
 * @property DateTime $suspendedUntil Время, когда заканчивается период охлаждения кредита или рассрочки.
 */
class PaymentMethodSberLoan extends AbstractPaymentMethod
{
    /**
     * Тариф кредита, который пользователь выбрал при оплате.
     *
     * Возможные значения:
     * * loan — кредит;
     * * installments_XX — рассрочка, где XX — количество месяцев для выплаты рассрочки. Например, installments_3 — рассрочка на 3 месяца.
     *
     * Присутствует для платежей в статусе `waiting_for_capture` и `succeeded`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^loan|installments_([0-9]{1,2})$/")]
    private ?string $_loan_option = null;

    /**
     * Сумма скидки для рассрочки.
     *
     * Присутствует для платежей в статусе `waiting_for_capture` и `succeeded`, если пользователь выбрал рассрочку.
     *
     * @var AmountInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_discount_amount = null;

    /**
     * Время, когда заканчивается [период охлаждения](https://yookassa.ru/docs/support/payments/credit-purchases-by-sberbank-with-cooling-off) кредита или рассрочки.
     * Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
     * Присутствует для платежей в статусе ~`pending`, которые по [закону](https://www.consultant.ru/document/cons_doc_LAW_498604/) попадают под процедуру охлаждения.
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_suspended_until = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::SBER_LOAN);
    }

    /**
     * Возвращает тариф кредита.
     */
    public function getLoanOption(): ?string
    {
        return $this->_loan_option;
    }

    /**
     * Устанавливает тариф кредита.
     */
    public function setLoanOption(?string $loan_option): self
    {
        $this->_loan_option = $this->validatePropertyValue('_loan_option', $loan_option);
        return $this;
    }

    /**
     * Возвращает сумму скидки для рассрочки.
     *
     * @return AmountInterface|null Сумма скидки для рассрочки
     */
    public function getDiscountAmount(): ?AmountInterface
    {
        return $this->_discount_amount;
    }

    /**
     * Устанавливает сумму скидки для рассрочки.
     *
     * @param AmountInterface|array|null $discount_amount Сумма скидки для рассрочки
     *
     * @return self
     */
    public function setDiscountAmount(mixed $discount_amount = null): self
    {
        $this->_discount_amount = $this->validatePropertyValue('_discount_amount', $discount_amount);
        return $this;
    }

    /**
     * Возвращает время, когда заканчивается период охлаждения кредита или рассрочки.
     *
     * @return DateTime|null
     */
    public function getSuspendedUntil(): ?DateTime
    {
        return $this->_suspended_until;
    }

    /**
     * Устанавливает время, когда заканчивается период охлаждения кредита или рассрочки.
     *
     * @param DateTime|string|null $suspended_until Время, когда заканчивается период охлаждения кредита или рассрочки.
     *
     * @return self
     */
    public function setSuspendedUntil(DateTime|string|null $suspended_until = null): self
    {
        $this->_suspended_until = $this->validatePropertyValue('_suspended_until', $suspended_until);
        return $this;
    }
}

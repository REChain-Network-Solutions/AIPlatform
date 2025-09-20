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

namespace YooKassa\Request\Invoices;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Payment\PaymentInterface;
use YooKassa\Model\Payment\Recipient;
use YooKassa\Model\Payment\RecipientInterface;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentData.
 *
 * Данные для проведения платежа по выставленному счету.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Сумма платежа. Должна укладываться в [лимиты](https://yookassa.ru/docs/support/payments/limits).
 * @property ReceiptInterface $receipt Данные для формирования чека.
 * @property RecipientInterface $recipient Получатель платежа. Нужен, если вы разделяете потоки платежей в рамках одного аккаунта или создаете платеж в адрес другого аккаунта.
 * @property bool $save_payment_method Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments).
 * @property bool $savePaymentMethod Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments).
 * @property bool $capture [Автоматический прием](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#capture-true) поступившего платежа.
 * @property string $client_ip IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
 * @property string $clientIp IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
 * @property string $description Описание транзакции (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь — при оплате. Например: «Оплата заказа № 72 для user@yoomoney.ru».
 * @property array $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
*/
class PaymentData extends AbstractObject
{
    /**
     * Сумма платежа. Должна укладываться в [лимиты](https://yookassa.ru/docs/support/payments/limits).
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Данные для формирования чека.
     * Необходимо передавать в этих случаях:
     * - вы компания или ИП и для оплаты с соблюдением требований 54-ФЗ используете [Чеки от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics);
     * - вы компания или ИП, для оплаты с соблюдением требований 54-ФЗ используете [стороннюю онлайн-кассу](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) и отправляете данные для чеков по одному из сценариев: [Платеж и чек одновременно](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics#payment-and-receipt) или [Сначала чек, потом платеж](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics#payment-after-receipt);
     * - вы самозанятый и используете решение ЮKassa для [автоотправки чеков](https://yookassa.ru/developers/payment-acceptance/receipts/self-employed/basics).
     *
     * @var ReceiptInterface|null
     */
    #[Assert\Type(Receipt::class)]
    private ?ReceiptInterface $_receipt = null;

    /**
     * Получатель платежа.
     * Нужен, если вы разделяете потоки платежей в рамках одного аккаунта или создаете платеж в адрес другого аккаунта.
     *
     * @var Recipient|null
     */
    #[Assert\Type(Recipient::class)]
    private ?RecipientInterface $_recipient = null;

    /**
     * Сохранение платежных данных для проведения [автоплатежей](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/recurring-payments).
     * Возможные значения:
     * - `true` — сохранить способ оплаты (сохранить платежные данные);
     * - `false` — провести платеж без сохранения способа оплаты.
     *
     * Доступно только после согласования с менеджером ЮKassa.
     *
     * @var bool|null
     */
    #[Assert\Type('bool')]
    private ?bool $_save_payment_method = null;

    /**
     * [Автоматический прием](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#capture-true) поступившего платежа.
     * Возможные значения:
     *  - `true` — оплата списывается сразу (платеж в одну стадию);
     * - `false` — оплата холдируется и списывается по вашему запросу ([платеж в две стадии](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#capture-and-cancel)).
     *
     * По умолчанию ~`false`.
     *
     * @var bool|null
     */
    #[Assert\Type('bool')]
    private ?bool $_capture = null;

    /**
     * IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_client_ip = null;

    /**
     * Описание транзакции (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь — при оплате. Например: «Оплата заказа № 72 для user@yoomoney.ru».
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: PaymentInterface::MAX_LENGTH_DESCRIPTION)]
    private ?string $_description = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * Возвращает amount.
     *
     * @return AmountInterface|null Сумма платежа
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает amount.
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

    /**
     * Возвращает receipt.
     *
     * @return ReceiptInterface|null Данные для формирования чека
     */
    public function getReceipt(): ?ReceiptInterface
    {
        return $this->_receipt;
    }

    /**
     * Устанавливает receipt.
     *
     * @param ReceiptInterface|array|null $receipt Данные для формирования чека
     *
     * @return self
     */
    public function setReceipt(mixed $receipt = null): self
    {
        $this->_receipt = $this->validatePropertyValue('_receipt', $receipt);
        return $this;
    }

    /**
     * Возвращает recipient.
     *
     * @return Recipient|null Получатель платежа
     */
    public function getRecipient(): ?Recipient
    {
        return $this->_recipient;
    }

    /**
     * Устанавливает recipient.
     *
     * @param Recipient|array|null $recipient Получатель платежа
     *
     * @return self
     */
    public function setRecipient(mixed $recipient = null): self
    {
        $this->_recipient = $this->validatePropertyValue('_recipient', $recipient);
        return $this;
    }

    /**
     * Возвращает save_payment_method.
     *
     * @return bool|null Сохранение платежных данных для проведения автоплатежей
     */
    public function getSavePaymentMethod(): ?bool
    {
        return $this->_save_payment_method;
    }

    /**
     * Устанавливает save_payment_method.
     *
     * @param bool|array|null $save_payment_method Сохранение платежных данных для проведения автоплатежей
     *
     * @return self
     */
    public function setSavePaymentMethod(mixed $save_payment_method = null): self
    {
        $this->_save_payment_method = $this->validatePropertyValue('_save_payment_method', $save_payment_method);
        return $this;
    }

    /**
     * Возвращает capture.
     *
     * @return bool|null Автоматический прием поступившего платежа
     */
    public function getCapture(): ?bool
    {
        return $this->_capture;
    }

    /**
     * Устанавливает capture.
     *
     * @param bool|array|null $capture Автоматический прием поступившего платежа
     *
     * @return self
     */
    public function setCapture(mixed $capture = null): self
    {
        $this->_capture = $this->validatePropertyValue('_capture', $capture);
        return $this;
    }

    /**
     * Возвращает client_ip.
     *
     * @return string|null IPv4 или IPv6-адрес пользователя
     */
    public function getClientIp(): ?string
    {
        return $this->_client_ip;
    }

    /**
     * Устанавливает client_ip.
     *
     * @param string|null $client_ip IPv4 или IPv6-адрес пользователя
     *
     * @return self
     */
    public function setClientIp(?string $client_ip = null): self
    {
        $this->_client_ip = $this->validatePropertyValue('_client_ip', $client_ip);
        return $this;
    }

    /**
     * Возвращает description.
     *
     * @return string|null Описание транзакции
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает description.
     *
     * @param string|null $description Описание транзакции
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

}

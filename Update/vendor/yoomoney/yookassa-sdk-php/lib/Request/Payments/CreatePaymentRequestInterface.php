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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Deal\PaymentDealInfo;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\RecipientInterface;
use YooKassa\Model\Payment\TransferInterface;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes;
use YooKassa\Request\Payments\PaymentData\AbstractPaymentData;
use YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder;
use YooKassa\Request\Payments\ReceiverData\AbstractReceiver;

/**
 * Interface CreatePaymentRequestInterface.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property Recipient $recipient Получатель платежа, если задан
 * @property AmountInterface $amount Сумма создаваемого платежа
 * @property string $description Описание транзакции
 * @property ReceiptInterface $receipt Данные фискального чека 54-ФЗ
 * @property string $paymentToken Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
 * @property string $payment_token Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
 * @property string $paymentMethodId Идентификатор записи о сохраненных платежных данных покупателя
 * @property string $payment_method_id Идентификатор записи о сохраненных платежных данных покупателя
 * @property AbstractPaymentData $paymentMethodData Данные используемые для создания метода оплаты
 * @property AbstractPaymentData $payment_method_data Данные используемые для создания метода оплаты
 * @property AbstractConfirmationAttributes $confirmation Способ подтверждения платежа
 * @property bool|null $savePaymentMethod Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method
 * @property bool|null $save_payment_method Сохранить платежные данные для последующего использования. Значение true инициирует создание многоразового payment_method
 * @property bool $capture Автоматически принять поступившую оплату
 * @property string $clientIp IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения
 * @property string $client_ip IPv4 или IPv6-адрес покупателя. Если не указан, используется IP-адрес TCP-подключения
 * @property Metadata $metadata Метаданные привязанные к платежу
 * @property PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
 * @property string $merchantCustomerId Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property string $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона
 * @property AbstractReceiver|null $receiver Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона
 */
interface CreatePaymentRequestInterface
{
    /**
     * Возвращает объект получателя платежа.
     *
     * @return null|Recipient Объект с информацией о получателе платежа или null, если получатель не задан
     */
    public function getRecipient(): ?Recipient;

    /**
     * Проверяет наличие получателя платежа в запросе.
     *
     * @return bool True если получатель платежа задан, false если нет
     */
    public function hasRecipient(): bool;

    /**
     * Устанавливает объект с информацией о получателе платежа.
     *
     * @param null|Recipient $recipient Инстанс объекта информации о получателе платежа или null
     */
    public function setRecipient(?Recipient $recipient);

    /**
     * Возвращает сумму заказа.
     *
     * @return AmountInterface|null Сумма заказа
     */
    public function getAmount(): ?AmountInterface;

    /**
     * Возвращает описание транзакции.
     *
     * @return string|null Описание транзакции
     */
    public function getDescription(): ?string;

    /**
     * Проверяет наличие описания транзакции в создаваемом платеже.
     *
     * @return bool True если описание транзакции установлено, false если нет
     */
    public function hasDescription(): bool;

    /**
     * Устанавливает описание транзакции.
     *
     * @param string|null $description Описание транзакции
     */
    public function setDescription(?string $description): self;

    /**
     * Возвращает чек, если он есть.
     *
     * @return null|ReceiptInterface Данные фискального чека 54-ФЗ или null, если чека нет
     */
    public function getReceipt(): ?ReceiptInterface;

    /**
     * Проверяет наличие чека в создаваемом платеже.
     *
     * @return bool True если чек есть, false если нет
     */
    public function hasReceipt(): bool;

    /**
     * Возвращает одноразовый токен для проведения оплаты.
     *
     * @return string|null Одноразовый токен для проведения оплаты, сформированный YooKassa JS widget
     */
    public function getPaymentToken(): ?string;

    /**
     * Проверяет наличие одноразового токена для проведения оплаты.
     *
     * @return bool True если токен установлен, false если нет
     */
    public function hasPaymentToken(): bool;

    /**
     * Устанавливает одноразовый токен для проведения оплаты, сформированный YooKassa JS widget.
     *
     * @param string|null $payment_token Одноразовый токен для проведения оплаты
     */
    public function setPaymentToken(?string $payment_token): self;

    /**
     * Устанавливает идентификатор записи платёжных данных покупателя.
     *
     * @return string|null Идентификатор записи о сохраненных платежных данных покупателя
     */
    public function getPaymentMethodId(): ?string;

    /**
     * Проверяет наличие идентификатора записи о платёжных данных покупателя.
     *
     * @return bool True если идентификатор задан, false если нет
     */
    public function hasPaymentMethodId(): bool;

    /**
     * Устанавливает идентификатор записи о сохранённых данных покупателя.
     *
     * @param string|null $payment_method_id Идентификатор записи о сохраненных платежных данных покупателя
     */
    public function setPaymentMethodId(?string $payment_method_id): self;

    /**
     * Возвращает данные для создания метода оплаты.
     *
     * @return AbstractPaymentData|null Данные используемые для создания метода оплаты
     */
    public function getPaymentMethodData(): ?AbstractPaymentData;

    /**
     * Проверяет установлен ли объект с методом оплаты.
     *
     * @return bool True если объект метода оплаты установлен, false если нет
     */
    public function hasPaymentMethodData(): bool;

    /**
     * Устанавливает объект с информацией для создания метода оплаты.
     *
     * @param null|AbstractPaymentData $payment_method_data Объект создания метода оплаты или null
     */
    public function setPaymentMethodData(?AbstractPaymentData $payment_method_data): self;

    /**
     * Возвращает способ подтверждения платежа.
     *
     * @return AbstractConfirmationAttributes|null Способ подтверждения платежа
     */
    public function getConfirmation(): ?AbstractConfirmationAttributes;

    /**
     * Проверяет, был ли установлен способ подтверждения платежа.
     *
     * @return bool True если способ подтверждения платежа был установлен, false если нет
     */
    public function hasConfirmation(): bool;

    /**
     * Устанавливает способ подтверждения платежа.
     *
     * @param null|array|AbstractConfirmationAttributes $confirmation Способ подтверждения платежа
     */
    public function setConfirmation(mixed $confirmation): self;

    /**
     * Возвращает флаг сохранения платёжных данных.
     *
     * @return bool|null Флаг сохранения платёжных данных
     */
    public function getSavePaymentMethod(): ?bool;

    /**
     * Проверяет, был ли установлен флаг сохранения платёжных данных.
     *
     * @return bool True если флыг был установлен, false если нет
     */
    public function hasSavePaymentMethod(): bool;

    /**
     * Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method.
     *
     * @param bool|null $save_payment_method Сохранить платежные данные для последующего использования
     */
    public function setSavePaymentMethod(?bool $save_payment_method = null): self;

    /**
     * Возвращает флаг автоматического принятия поступившей оплаты.
     *
     * @return bool True если требуется автоматически принять поступившую оплату, false если нет
     */
    public function getCapture(): bool;

    /**
     * Проверяет, был ли установлен флаг автоматического приняти поступившей оплаты.
     *
     * @return bool True если флаг автоматического принятия оплаты был установлен, false если нет
     */
    public function hasCapture(): bool;

    /**
     * Устанавливает флаг автоматического принятия поступившей оплаты.
     *
     * @param bool $capture Автоматически принять поступившую оплату
     */
    public function setCapture(bool $capture): self;

    /**
     * Возвращает IPv4 или IPv6-адрес покупателя.
     *
     * @return string|null IPv4 или IPv6-адрес покупателя
     */
    public function getClientIp(): ?string;

    /**
     * Проверяет, был ли установлен IPv4 или IPv6-адрес покупателя.
     *
     * @return bool True если IP адрес покупателя был установлен, false если нет
     */
    public function hasClientIp(): bool;

    /**
     * Устанавливает IP адрес покупателя.
     *
     * @param string|null $client_ip IPv4 или IPv6-адрес покупателя
     */
    public function setClientIp(?string $client_ip): self;

    /**
     * Возвращает данные оплаты установленные мерчантом
     *
     * @return Metadata|null Метаданные привязанные к платежу
     */
    public function getMetadata(): ?Metadata;

    /**
     * Проверяет, были ли установлены метаданные заказа.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool;

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param null|array|Metadata $metadata Метаданные платежа, устанавливаемые мерчантом
     */
    public function setMetadata(mixed $metadata): self;

    /**
     * Возвращает данные длинной записи.
     */
    public function getAirline(): ?AirlineInterface;

    /**
     * Проверяет, были ли установлены данные длинной записи.
     */
    public function hasAirline(): bool;

    /**
     * Устанавливает данные авиабилетов.
     *
     * @param AirlineInterface|array|null $airline Данные авиабилетов
     */
    public function setAirline(mixed $airline): AbstractRequestInterface;

    /**
     * Проверяет наличие данных о распределении денег.
     */
    public function hasTransfers(): bool;

    /**
     * Возвращает данные о распределении денег — сколько и в какой магазин нужно перевести.
     * Присутствует, если вы используете решение ЮKassa для платформ.
     * (https://yookassa.ru/developers/special-solutions/checkout-for-platforms/basics).
     *
     * @return TransferInterface[]|ListObjectInterface|null Данные о распределении денег
     */
    public function getTransfers(): ?ListObjectInterface;

    /**
     * Устанавливает данные о распределении денег — сколько и в какой магазин нужно перевести.
     * Присутствует, если вы используете решение ЮKassa для платформ.
     * (https://yookassa.ru/developers/special-solutions/checkout-for-platforms/basics).
     *
     * @param ListObjectInterface|array|null $transfers Массив распределения денег
     *
     * @return self
     */
    public function setTransfers(mixed $transfers = null): AbstractRequestInterface;

    /**
     * Возвращает данные о сделке, в составе которой проходит платеж.
     *
     * @return PaymentDealInfo|null Данные о сделке, в составе которой проходит платеж
     */
    public function getDeal(): ?PaymentDealInfo;

    /**
     * Проверяет, были ли установлены данные о сделке.
     *
     * @return bool True если данные о сделке были установлены, false если нет
     */
    public function hasDeal(): bool;

    /**
     * Устанавливает данные о сделке, в составе которой проходит платеж.
     *
     * @param null|array|PaymentDealInfo $deal Данные о сделке, в составе которой проходит платеж
     */
    public function setDeal(mixed $deal): self;

    /**
     * Возвращает идентификатор покупателя в вашей системе.
     *
     * @return string|null Идентификатор покупателя в вашей системе
     */
    public function getMerchantCustomerId(): ?string;

    /**
     * Проверяет, был ли установлен идентификатор покупателя в вашей системе.
     *
     * @return bool True если идентификатор покупателя был установлен, false если нет
     */
    public function hasMerchantCustomerId(): bool;

    /**
     * Устанавливает идентификатор покупателя в вашей системе.
     *
     * @param string|null $merchant_customer_id Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов
     */
    public function setMerchantCustomerId(?string $merchant_customer_id): self;

    /**
     * Возвращает платежное поручение.
     *
     * @return AbstractPaymentOrder|null Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ.
     */
    public function getPaymentOrder(): ?AbstractPaymentOrder;

    /**
     * Проверяет, было ли установлено платежное поручение.
     *
     * @return bool True если платежное поручение было установлены, false если нет
     */
    public function hasPaymentOrder(): bool;

    /**
     * Устанавливает платежное поручение.
     *
     * @param AbstractPaymentOrder|array|null $payment_order Платежное поручение — распоряжение на перевод банку для оплаты жилищно-коммунальных услуг (ЖКУ), сведения о платеже для регистрации в ГИС ЖКХ.
     */
    public function setPaymentOrder(mixed $payment_order = null): self;

    /**
     * Возвращает реквизиты получателя оплаты.
     *
     * @return null|AbstractReceiver Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона.
     */
    public function getReceiver(): ?AbstractReceiver;

    /**
     * Проверяет, были ли установлены реквизиты получателя оплаты.
     *
     * @return bool True если реквизиты получателя оплаты были установлены, false если нет
     */
    public function hasReceiver(): bool;

    /**
     * Устанавливает реквизиты получателя оплаты.
     *
     * @param null|array|AbstractReceiver $receiver Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона.
     */
    public function setReceiver(mixed $receiver): self;
}

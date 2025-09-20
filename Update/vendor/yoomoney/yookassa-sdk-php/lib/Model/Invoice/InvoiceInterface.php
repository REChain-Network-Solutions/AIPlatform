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

use DateTime;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod;
use YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf;
use YooKassa\Model\Metadata;

/**
 * Класс, представляющий модель Invoice.
 *
 * Данные о счете.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор счета в ЮKassa.
 * @property string $status Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`.
 * @property LineItem[]|ListObjectInterface $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
 * @property AbstractDeliveryMethod|null $delivery_method Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.
 * @property AbstractDeliveryMethod|null $deliveryMethod Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.
 * @property PaymentDetails|null $payment_details Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).
 * @property PaymentDetails|null $paymentDetails Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).
 * @property DateTime $created_at Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z`
 * @property DateTime $createdAt Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z`
 * @property DateTime|null $expires_at Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`.
 * @property DateTime|null $expiresAt Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`.
 * @property string|null $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
 * @property InvoiceCancellationDetails|null $cancellation_details Комментарий к статусу `canceled`: кто отменил счет и по какой причине.
 * @property InvoiceCancellationDetails|null $cancellationDetails Комментарий к статусу `canceled`: кто отменил счет и по какой причине.
 * @property Metadata|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8./
 */
interface InvoiceInterface
{
    /** @var int Минимальная длина идентификатора */
    public const MIN_LENGTH_ID = 39;

    /** @var int Максимальная длина идентификатора */
    public const MAX_LENGTH_ID = 39;

    /** @var int Максимальная длина описания счета */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /**
     * Возвращает id.
     *
     * @return string|null
     */
    public function getId(): ?string;

    /**
     * Устанавливает id.
     *
     * @param string|null $id Идентификатор счета в ЮKassa.
     *
     * @return self
     */
    public function setId(?string $id = null): self;

    /**
     * Возвращает status.
     *
     * @return string|null
     */
    public function getStatus(): ?string;

    /**
     * Устанавливает status.
     *
     * @param string|null $status
     *
     * @return self
     */
    public function setStatus(?string $status = null): self;

    /**
     * Возвращает cart.
     *
     * @return LineItem[]|ListObjectInterface|null
     */
    public function getCart(): ?ListObjectInterface;

    /**
     * Устанавливает cart.
     *
     * @param ListObjectInterface|array|null $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
     *
     * @return self
     */
    public function setCart(mixed $cart = null): self;

    /**
     * Возвращает delivery_method.
     *
     * @return AbstractDeliveryMethod|DeliveryMethodSelf|null
     */
    public function getDeliveryMethod(): ?AbstractDeliveryMethod;

    /**
     * Устанавливает delivery_method.
     *
     * @param AbstractDeliveryMethod|array|null $delivery_method
     *
     * @return self
     */
    public function setDeliveryMethod(mixed $delivery_method = null): self;

    /**
     * Возвращает payment_details.
     *
     * @return PaymentDetails|null Данные о платеже по выставленному счету
     */
    public function getPaymentDetails(): ?PaymentDetails;

    /**
     * Устанавливает payment_details.
     *
     * @param PaymentDetails|array|null $payment_details Данные о платеже по выставленному счету
     *
     * @return self
     */
    public function setPaymentDetails(mixed $payment_details = null): self;

    /**
     * Возвращает created_at.
     *
     * @return DateTime|null Дата и время создания счета на оплату
     */
    public function getCreatedAt(): ?DateTime;

    /**
     * Устанавливает created_at.
     *
     * @param DateTime|string|null $created_at Дата и время создания счета на оплату
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at): self;

    /**
     * Возвращает expires_at.
     *
     * @return DateTime|null Срок действия счета
     */
    public function getExpiresAt(): ?DateTime;

    /**
     * Устанавливает expires_at.
     *
     * @param DateTime|string|null $expires_at Срок действия счета
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self;

    /**
     * Возвращает description.
     *
     * @return string|null Описание выставленного счета
     */
    public function getDescription(): ?string;

    /**
     * Устанавливает description.
     *
     * @param string|null $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
     *
     * @return self
     */
    public function setDescription(?string $description = null): self;

    /**
     * Возвращает cancellation_details.
     *
     * @return InvoiceCancellationDetails|null Комментарий к статусу `canceled`
     */
    public function getCancellationDetails(): ?InvoiceCancellationDetails;

    /**
     * Устанавливает cancellation_details.
     *
     * @param InvoiceCancellationDetails|array|null $cancellation_details Комментарий к статусу `canceled`
     *
     * @return self
     */
    public function setCancellationDetails(mixed $cancellation_details = null): self;

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata;

    /**
     * Устанавливает metadata.
     *
     * @param string|array|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self;
}

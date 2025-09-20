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


use DateTime;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Invoice\LineItem;
use YooKassa\Model\Metadata;
use YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData;

/**
 * Класс, представляющий модель CreateInvoiceRequest.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property PaymentData $payment_data Данные для проведения платежа по выставленному счету.
 * @property LineItem[]|ListObjectInterface $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
 * @property AbstractDeliveryMethodData $delivery_method_data Данные о способе доставки счета пользователю. Доступен только один способ — самостоятельная доставка: ЮKassa возвращает вам ссылку на счет, и вы передаете ее пользователю любым удобным для вас способом.
 * @property DateTime $expires_at Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`
 * @property string $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен.
 * @property string $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
 */
interface CreateInvoiceRequestInterface extends AbstractRequestInterface
{
    /**
     * Возвращает payment_data.
     *
     * @return PaymentData|null Данные для проведения платежа по выставленному счету
     */
    public function getPaymentData(): ?PaymentData;

    /**
     * Устанавливает payment_data.
     *
     * @param PaymentData|array|null $payment_data Данные для проведения платежа по выставленному счету
     *
     * @return CreateInvoiceRequestInterface
     */
    public function setPaymentData(mixed $payment_data = null): CreateInvoiceRequestInterface;

    /**
     * Возвращает cart.
     *
     * @return LineItem[]|ListObjectInterface Корзина заказа — список товаров или услуг
     */
    public function getCart(): ListObjectInterface;

    /**
     * Устанавливает cart.
     *
     * @param ListObjectInterface|array|null $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
     *
     * @return CreateInvoiceRequestInterface
     */
    public function setCart(mixed $cart = null): CreateInvoiceRequestInterface;

    /**
     * Возвращает delivery_method_data.
     *
     * @return AbstractDeliveryMethodData|null Данные о способе доставки счета пользователю
     */
    public function getDeliveryMethodData(): ?AbstractDeliveryMethodData;

    /**
     * Устанавливает delivery_method_data.
     *
     * @param AbstractDeliveryMethodData|array|null $delivery_method_data Данные о способе доставки счета пользователю
     *
     * @return CreateInvoiceRequestInterface
     */
    public function setDeliveryMethodData(mixed $delivery_method_data = null): CreateInvoiceRequestInterface;

    /**
     * Возвращает expires_at.
     *
     * @return DateTime|null Срок действия счета
     */
    public function getExpiresAt(): ?DateTime;

    /**
     * Устанавливает expires_at.
     *
     * @param DateTime|string|null $expires_at Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`
     *
     * @return CreateInvoiceRequestInterface
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): CreateInvoiceRequestInterface;

    /**
     * Возвращает locale.
     *
     * @return string|null Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
     */
    public function getLocale(): ?string;

    /**
     * Устанавливает locale.
     *
     * @param string|null $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
     *
     * @return CreateInvoiceRequestInterface
     */
    public function setLocale(?string $locale = null): CreateInvoiceRequestInterface;

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
     * @return CreateInvoiceRequestInterface
     */
    public function setDescription(?string $description = null): CreateInvoiceRequestInterface;

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata;

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @return CreateInvoiceRequestInterface
     */
    public function setMetadata(mixed $metadata = null): CreateInvoiceRequestInterface;
}

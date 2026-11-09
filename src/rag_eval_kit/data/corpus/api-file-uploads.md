# File Uploads

> Synthetic documentation for Larkspur, a fictional project-management SaaS. Not a real product.

## Size limits

Attachments can be up to 100 MB each on paid plans. On the Free plan the limit is 25 MB per file. Executable files such as .exe and .msi are blocked.

## Uploading through the API

Small files can be sent directly with a multipart/form-data request to the attachments endpoint. For files larger than 10 MB, request a presigned upload URL first and upload the file directly to storage. Presigned URLs expire after 15 minutes.
